"""CPU-only recount and provenance audit of the completed 16M extension."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = Path('/mnt/raid0/RecursiveCompressor/experiments/logkv-causal-conv-20260914')
OUT = ROOT / HERE.name


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    pre = json.loads((OUT / 'preflight.json').read_text())
    full = json.loads((OUT / 'full.json').read_text())
    launch = json.loads((OUT / 'launch.json').read_text())
    assert pre['passed'] and full['passed']
    assert full['provenance'] == pre['provenance']
    provenance = full['provenance']
    assert sha(HERE / 'evaluate.py') == provenance['script_sha256']
    assert sha(OUT / 'preflight.json') == launch['preflight_sha256']
    assert launch['gpus'] == [0] and launch['projected_full_seconds_with_margin'] < 8*3600
    assert full['elapsed_seconds'] < launch['timeout_seconds']
    manifest_path = HERE.parent / 'source_manifest.json'
    assert sha(manifest_path) == provenance['source_manifest_sha256']
    for name, digest in json.loads(manifest_path.read_text())['files'].items():
        assert sha(ROOT / 'source' / name) == digest
    checkpoint = Path(provenance['checkpoint'])
    assert sha(checkpoint / 'config.json') == provenance['config_sha256']
    for folder in ['model', 'model_best']:
        assert sha(checkpoint.parent / folder / 'model.safetensors') == provenance['checkpoint_sha256']
    expected = torch.randint(1, 9, (8, 10), generator=torch.Generator().manual_seed(12345)).numpy()
    assert len({tuple(r) for r in expected.tolist()}) == 8
    rows = []
    for payload, horizons in [(pre, [2**17, 2**20]), (full, [2**24])]:
        assert [r['T'] for r in payload['rows']] == horizons
        for row in payload['rows']:
            assert row['n'] == 8 and row['batch_size'] == 4 and row['chunk_len'] == 8192
            assert row['seed'] == 12345 and row['sequence_length'] == row['T'] + 20
            arrays = {key: np.asarray([v for rec in row['records'] for v in rec[key]])
                      for key in ['memory', 'target', 'prediction', 'margin', 'logits']}
            assert len(row['records']) == 2
            assert np.array_equal(arrays['memory'], expected)
            assert np.array_equal(arrays['target'], expected)
            logits = arrays['logits']
            assert logits.shape == (8, 10, 10) and np.isfinite(logits).all()
            pred = logits.argmax(-1)
            assert np.array_equal(pred, arrays['prediction'])
            correct = np.take_along_axis(logits, expected[..., None], -1).squeeze(-1)
            rivals = logits.copy()
            np.put_along_axis(rivals, expected[..., None], -np.inf, -1)
            margins = correct-rivals.max(-1)
            assert np.array_equal(margins, arrays['margin'])
            exact = (pred == expected).all(-1)
            tokens = (pred == expected)
            assert int(exact.sum()) == row['string_correct']
            assert int(tokens.sum()) == row['token_correct']
            rows.append(dict(T=row['T'], n=8, string_correct=int(exact.sum()), token_correct=int(tokens.sum()),
                digit_errors=(~tokens).sum(0).tolist(), minimum_margin=float(margins.min()),
                median_margin=float(np.median(margins)), seconds=row['seconds']))
    assert pre['rows'][0]['standard_evaluator_counts_match']
    main_commit = subprocess.check_output(['git', 'rev-parse', 'main'], text=True).strip()
    assert main_commit == 'd707675bd2dee14f19a46a5f1d686b3ec9c11326'
    files = ['preflight.json', 'full.json', 'launch.json', 'preflight.log', 'full.log']
    for name in files:
        shutil.copy2(OUT / name, HERE / name)
    review = dict(passed=True, cpu_only=True, main_unchanged=main_commit, rows=rows,
        checkpoint_and_frozen_source_hashes_verified=True, seed_replay_verified=True,
        saved_logits_recounted=True, script_sha256=sha(Path(__file__)),
        files={name: sha(HERE / name) for name in files},
        limitations='One trained checkpoint, eight fixed random strings shared across the three horizons; not 256 examples, not all intermediate T, variable memory length, or Selective Copying.')
    (HERE / 'review.json').write_text(json.dumps(review, indent=2, allow_nan=False)+'\n')
    print(json.dumps(review, indent=2))


if __name__ == '__main__':
    main()
