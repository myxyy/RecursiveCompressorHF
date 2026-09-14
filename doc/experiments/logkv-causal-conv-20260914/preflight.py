"""Verify reused Copying initialization, deterministic task repeats, and timing."""
import json
import os
import subprocess
import sys
import time

from common import HERE, INITIAL, BASE_INITIAL, MODES, ROOT, SOURCE, GPUS, NUM_LAYERS, NUM_PARAMS, command, now, save, sha

sys.path.insert(0, str(SOURCE))

import torch
from configuration_logkv import LogKVConfig
from logkv_lm import LogKVLM
from safetensors.torch import load_file


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    prior = json.loads((ROOT / 'source_manifest.json').read_text())
    for file, digest in prior['files'].items():
        assert sha(SOURCE / file) == digest
    torch.manual_seed(0)
    config = LogKVConfig(vocab_size=10, d_model=512, num_heads=8, d_ff=1024,
        num_layers=NUM_LAYERS, conv_kernel_size=4, chunk_size=4, phase_emb=False, phase_levels=2,
        gated_attention=True, self_slot=True, pad_token_id=None, bos_token_id=None,
        eos_token_id=None)
    model = LogKVLM(config)
    assert sum(p.numel() for p in model.parameters()) == NUM_PARAMS
    assert not any('phase_emb' in key for key in model.state_dict())
    state = model.state_dict(); base = load_file(BASE_INITIAL)
    assert all(key in state and state[key].shape == value.shape for key, value in base.items())
    extra = set(state) - set(base)
    assert extra and all('.causal_conv.' in key for key in extra)
    fresh_extra = {key: state[key].clone() for key in extra}
    state.update(base); model.load_state_dict(state, strict=True)
    assert all(torch.equal(model.state_dict()[key], value) for key, value in base.items())
    assert all(torch.equal(model.state_dict()[key], value) for key, value in fresh_extra.items())
    assert not INITIAL.exists()
    model.save_pretrained(INITIAL.parent)
    record = dict(passed=False, tasks=list(MODES), initial_weights_identical=True,
        num_params=NUM_PARAMS, num_layers=NUM_LAYERS, conv_kernel_size=4, shared_two_layer_initial_weights_identical=True,
        base_initial_sha256=sha(BASE_INITIAL), extra_conv_parameters=sum(v.numel() for v in fresh_extra.values()), initial_weights_path=str(INITIAL),
        initial_weights_sha256=sha(INITIAL), deterministic_algorithms=True,
        cublas_workspace_config=':4096:8', commands=[])
    record['gpu_started_unix'] = time.time()
    record['gpu_started'] = now()

    def batch(label, modes, steps):
        processes = []
        logs = []
        try:
            for gpu, mode in zip(GPUS, modes):
                run_name = f'{label}-gpu{gpu}-{mode}'
                cmd = command(mode, sys.executable)
                for flag, value in [('--run-name', run_name), ('--steps', str(steps))]:
                    cmd[cmd.index(flag) + 1] = value
                cmd += ['--eval-interval', '0', '--log-interval', str(min(steps, 100)),
                        '--save-interval', str(steps)]
                data = ROOT / 'preflight-training'
                assert not (data / 'exp' / mode / run_name).exists()
                env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'DATA_DIR': str(data), 'LOGKV_TASK': mode,
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
        return [ROOT / 'preflight-training/exp' / mode / f'{label}-gpu{gpu}-{mode}'
                for gpu, mode in zip(GPUS, modes)]

    try:
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(GPUS[0]), 'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
            'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}
        with (ROOT / 'fullsize_smoke.log').open('w') as log:
            subprocess.run([sys.executable, str(HERE / 'fullsize_smoke.py')], cwd=SOURCE,
                env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        record['fullsize_smoke'] = json.loads((HERE / 'fullsize_smoke.json').read_text())
        assert record['fullsize_smoke']['passed']
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
                assert audit['passed'] and audit['task'] == mode
                assert audit['task_sha256'] == sha(SOURCE / 'exp' / mode / 'task.py')
                assert audit['initial_weights_sha256'] == record['initial_weights_sha256']
            repeats[mode] = dict(
                steps=20, physical_gpus=list(GPUS), weights_bitexact=True,
                logged_metrics_bitexact=True,
                sha256=[sha(path / 'model/model.safetensors') for path in paths])
            print(f'{mode}: 20-step task repeat bitexact across GPUs 0/1', flush=True)
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
        # 15% timing margin and one hour for the 164 standard evaluation cells.
        estimate = max(row['train_hours_projected'] for row in timing.values()) * 1.15
        estimate += 1.0 + gpu_preflight_hours
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
