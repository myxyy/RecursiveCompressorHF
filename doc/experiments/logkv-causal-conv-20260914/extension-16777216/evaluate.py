"""Copying-only 16M extension; use the original task and frozen trained model.

Run preflight first, inspect its estimate, then run full. Never retrain or merge.
All large input tensors stay transient; checkpoints are read-only on RAID.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import ROOT, SOURCE, run_dir, sha, save, now, bind_task

OUT = ROOT / 'extension-16777216'
MODEL = run_dir('copying') / 'model_best'
WEIGHT_SHA = '0a42b83130217e513af2aa4d1000f185d27ca0c522a807a92d4789982af80468'
T_MAX = 2**24
SAMPLES = 8
BATCH = 4
SEED = 12345
CHUNK = 8192


def audit():
    manifest = json.loads((HERE.parent / 'source_manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert sha(SOURCE / name) == digest, name
    for folder in ['model_best', 'model']:
        assert sha(run_dir('copying') / folder / 'model.safetensors') == WEIGHT_SHA
    config = json.loads((MODEL / 'config.json').read_text())
    assert config['num_layers'] == 2 and config['conv_kernel_size'] == 4
    assert not config['phase_emb'] and config['self_slot'] and config['gated_attention']
    return dict(source_commit=manifest['source_commit'], source_manifest_sha256=sha(HERE.parent / 'source_manifest.json'),
                script_sha256=sha(Path(__file__)), checkpoint=str(MODEL), checkpoint_sha256=WEIGHT_SHA,
                config_sha256=sha(MODEL / 'config.json'), best_final_same_weights=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['preflight', 'full'])
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'Only GPU0 is used by this extension'
    info = audit()
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / (args.stage + '.json')
    assert not output.exists(), f'Refusing to overwrite {output}'
    if args.stage == 'full':
        pre = json.loads((OUT / 'preflight.json').read_text())
        assert pre['passed'] and pre['provenance'] == info
        assert pre['projected_full_seconds_with_margin'] < 8 * 3600
    import torch
    sys.path.insert(0, str(SOURCE))
    task = bind_task('copying')
    from logkv_lm import LogKVLM
    spec = importlib.util.spec_from_file_location('standard_eval', SOURCE / 'exp/copying/evaluate.py')
    standard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(standard)
    assert standard.make_batch is task.make_batch and standard.CHUNK_LEN == CHUNK
    # Match the completed standard evaluation's precision and kernel settings.
    model = LogKVLM.from_pretrained(MODEL).to(device='cuda:0', dtype=torch.bfloat16).eval()
    started = now()
    started_mono = time.monotonic()
    rows = []

    @torch.no_grad()
    def evaluate(T, parity=False):
        generator = torch.Generator().manual_seed(SEED)
        records = []
        torch.cuda.synchronize()
        t0 = time.monotonic()
        for batch_index in range(SAMPLES // BATCH):
            inputs, labels = task.make_batch(T, BATCH, generator=generator, device='cuda:0')
            memory = inputs[:, :10].clone()
            assert torch.equal(labels[:, -10:], memory)
            assert inputs.shape == (BATCH, T + 20)
            assert (inputs[:, 10:T+9] == 0).all() and (inputs[:, T+9:] == 9).all()
            hidden = None
            last_report = time.monotonic()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                for i in range(0, T + 20, CHUNK):
                    logits, hidden = model.step(inputs[:, i:i+CHUNK], hidden)
                    if time.monotonic() - last_report >= 25:
                        torch.cuda.synchronize()
                        save(OUT / 'progress.json', dict(stage=args.stage, T=T, batch=batch_index+1,
                            batches=2, tokens_processed=min(i+CHUNK, T+20), sequence_length=T+20,
                            started=started, updated=now(), elapsed_seconds=time.monotonic()-started_mono))
                        print(f'T={T} batch={batch_index+1}/2 tokens={min(i+CHUNK,T+20)}/{T+20}', flush=True)
                        last_report = time.monotonic()
            answer = logits[:, -10:].float()
            assert torch.isfinite(answer).all()
            target = labels[:, -10:]
            pred = answer.argmax(-1)
            correct = answer.gather(-1, target[..., None]).squeeze(-1)
            rivals = answer.clone().scatter_(-1, target[..., None], float('-inf'))
            margin = correct - rivals.max(-1).values
            tok, st, tn, sn = task.score_logits(logits.float(), labels)
            assert tok == int((pred == memory).sum()) and st == int((pred == memory).all(-1).sum())
            assert (tn, sn) == (40, 4)
            records.append(dict(memory=memory.cpu().tolist(), target=target.cpu().tolist(),
                prediction=pred.cpu().tolist(), margin=margin.cpu().tolist(), logits=answer.cpu().tolist(),
                token_correct=tok, string_correct=st))
            del hidden, logits, inputs, labels, answer, memory, pred, target, correct, rivals, margin
        torch.cuda.synchronize()
        elapsed = time.monotonic()-t0
        result = dict(T=T, sequence_length=T+20, n=SAMPLES, seed=SEED, batch_size=BATCH,
            chunk_len=CHUNK, token_correct=sum(r['token_correct'] for r in records),
            string_correct=sum(r['string_correct'] for r in records), seconds=elapsed, records=records)
        if parity:
            # Independent invocation of the original evaluator, with identical RNG/batching.
            expected = standard.eval_horizon(model, T, SAMPLES, torch.Generator().manual_seed(SEED),
                                             torch.device('cuda:0'), True)
            # At T=131072 the original TOKEN_BUDGET chooses batch3, so compare counts,
            # while both paths use the same eight random memory strings and chunk boundaries.
            assert expected == (result['token_correct']/80, result['string_correct']/8)
            result['standard_evaluator_counts_match'] = True
        print(f'T={T}: exact={result["string_correct"]}/8 digits={result["token_correct"]}/80 seconds={elapsed:.2f}', flush=True)
        return result

    with torch.inference_mode():
        if args.stage == 'preflight':
            rows.append(evaluate(2**17, parity=True))
            rows.append(evaluate(2**20))
        else:
            rows.append(evaluate(T_MAX))
    assert audit() == info
    payload = dict(passed=True, stage=args.stage, started=started, completed=now(),
        elapsed_seconds=time.monotonic()-started_mono, provenance=info, precision='bf16 weights + bf16 autocast',
        gpu=0, gpu_name=torch.cuda.get_device_name(0), torch_version=torch.__version__,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        rows=rows, evaluation_only=True, task='copying', memory_len=10)
    if args.stage == 'preflight':
        payload['projected_full_seconds_with_margin'] = rows[-1]['seconds'] * 16 * 1.5 + 60
    save(output, payload)
    print(f'Saved {output}', flush=True)


if __name__ == '__main__':
    main()
