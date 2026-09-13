"""Verify reused Copying initialization, deterministic Selective repeats, and timing."""
import json
import os
import subprocess
import sys
import time

from common import (COPY_ROOT, HERE, INITIAL, MODES, ROOT, SOURCE, TASK_NAME,
                    bind_selective_task, command, now, save, sha, task_identity)

sys.path.insert(0, str(SOURCE))

import torch
from configuration_logkv import LogKVConfig
from logkv_lm import LogKVLM
from safetensors.torch import load_file


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    task = bind_selective_task()
    baseline = load_file(INITIAL)
    prior = json.loads((COPY_ROOT / 'source_manifest.json').read_text())
    for file, digest in prior['files'].items():
        assert sha(SOURCE / file) == digest
    save(ROOT / 'source_manifest.json', prior)
    initial_rng = None
    for mode in MODES:
        torch.manual_seed(0)
        config = LogKVConfig(
            vocab_size=10, d_model=512, num_heads=8, d_ff=1024, num_layers=2,
            chunk_size=4, phase_emb=False, phase_levels=2, gated_attention=True,
            self_slot=True, retrieval_rope=mode == 'local-control',
            aligned_rope=mode == 'aligned', pad_token_id=None, bos_token_id=None,
            eos_token_id=None)
        model = LogKVLM(config)
        assert sum(p.numel() for p in model.parameters()) == 5786112
        state = model.state_dict()
        assert state.keys() == baseline.keys()
        assert all(torch.equal(value, baseline[key]) for key, value in state.items())
        if initial_rng is None:
            initial_rng = torch.get_rng_state()
        else:
            assert torch.equal(initial_rng, torch.get_rng_state())
    # CPU-only task format check uses its own generator; it does not consume training RNG.
    inputs, labels = task.make_batch(100, 32, generator=torch.Generator().manual_seed(123))
    data_mask = (inputs >= 1) & (inputs <= 8)
    assert (data_mask.sum(-1) == 10).all()
    assert not data_mask[:, :10].all(dim=1).all()
    positions = data_mask.nonzero(as_tuple=False)[:, 1].reshape(32, 10)
    assert torch.equal(inputs.gather(1, positions), labels[:, -10:])
    record = dict(
        passed=False, initial_weights_identical=True, initial_rng_identical=True,
        num_params=5786112, initial_weights_path=str(INITIAL),
        initial_weights_sha256=sha(INITIAL), deterministic_algorithms=True,
        cublas_workspace_config=':4096:8', task_format_passed=True,
        commands=[], **task_identity())
    record['gpu_started_unix'] = time.time()
    record['gpu_started'] = now()

    def batch(label, modes, steps):
        processes = []
        logs = []
        try:
            for gpu, mode in enumerate(modes):
                run_name = f'{label}-gpu{gpu}-{mode}'
                cmd = command(mode, sys.executable)
                for flag, value in [('--run-name', run_name), ('--steps', str(steps))]:
                    cmd[cmd.index(flag) + 1] = value
                cmd += ['--eval-interval', '0', '--log-interval', str(min(steps, 100)),
                        '--save-interval', str(steps)]
                data = ROOT / 'preflight-training'
                assert not (data / 'exp' / TASK_NAME / run_name).exists()
                env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'DATA_DIR': str(data),
                       'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                       'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'PYTHONUNBUFFERED': '1',
                       'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}
                log = (ROOT / f'{run_name}.log').open('w')
                logs.append(log)
                process = subprocess.Popen(cmd, cwd=SOURCE, env=env, stdout=log,
                                           stderr=subprocess.STDOUT)
                processes.append(process)
                record['commands'].append(dict(label=label, mode=mode, gpu=gpu,
                                               command=cmd, run_name=run_name))
            for process in processes:
                code = process.wait(timeout=600)
                if code:
                    raise RuntimeError(f'Preflight failed; inspect {label} logs: exit {code}')
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            for log in logs:
                log.close()
        return [ROOT / 'preflight-training/exp' / TASK_NAME / f'{label}-gpu{gpu}-{mode}'
                for gpu, mode in enumerate(modes)]

    try:
        repeats = {}
        for mode in MODES:
            paths = batch('repeat-' + mode, [mode, mode], 20)
            first, second = [load_file(path / 'model/model.safetensors') for path in paths]
            assert first.keys() == second.keys()
            assert all(torch.equal(value, second[key]) for key, value in first.items())
            metrics = [[json.loads(line) for line in (path / 'train_log.jsonl').read_text().splitlines()]
                       for path in paths]
            strip_timing = lambda rows: [{key: value for key, value in row.items()
                                          if key != 'elapsed_sec'} for row in rows]
            assert strip_timing(metrics[0]) == strip_timing(metrics[1])
            for path in paths:
                audit = json.loads((path / 'initialization_audit.json').read_text())
                assert audit['passed'] and audit['task'] == TASK_NAME
                assert audit['task_sha256'] == record['task_sha256']
                assert audit['initial_weights_sha256'] == record['initial_weights_sha256']
            repeats[mode] = dict(
                steps=20, physical_gpus=[0, 1], weights_bitexact=True,
                logged_metrics_bitexact=True,
                sha256=[sha(path / 'model/model.safetensors') for path in paths])
            print(f'{mode}: 20-step Selective repeat bitexact across GPUs 0/1', flush=True)
        paths = batch('benchmark', MODES, 300)
        timing = {}
        for mode, path in zip(MODES, paths):
            rows = [json.loads(line) for line in (path / 'train_log.jsonl').read_text().splitlines()
                    if '"loss"' in line]
            assert rows[-1]['step'] == 300
            elapsed = rows[-1]['elapsed_sec']
            timing[mode] = dict(steps=300, elapsed_seconds=elapsed,
                                seconds_per_step=elapsed / 300,
                                train_hours_projected=elapsed / 300 * 50000 / 3600)
        gpu_preflight_hours = (time.time() - record['gpu_started_unix']) / 3600
        # 15% timing margin and half an hour for the 132 standard evaluation cells.
        estimate = max(row['train_hours_projected'] for row in timing.values()) * 1.15
        estimate += 0.5 + gpu_preflight_hours
        record.update(
            passed=True, repeat_training_bitexact=True, repeats=repeats, timing=timing,
            estimated_campaign_hours=estimate, gpu_preflight_hours=gpu_preflight_hours,
            finished=now(),
            limitation='20-step repeat does not prove 50k-step or cross-environment bitwise reproducibility.')
        assert sha(INITIAL) == record['initial_weights_sha256']
        print(json.dumps(record, indent=2), flush=True)
    except BaseException as exc:
        record.update(passed=False, error=str(exc), finished=now())
        raise
    finally:
        save(HERE / 'preflight.json', record)


if __name__ == '__main__':
    main()
