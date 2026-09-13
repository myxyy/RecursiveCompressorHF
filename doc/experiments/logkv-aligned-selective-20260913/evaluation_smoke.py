"""Actual Selective checkpoint evaluation and independent sample-stream replay."""
import json
import os
import shutil
import sys

import torch
import common

actual = common.ROOT
common.ROOT = actual / 'evaluation-smoke'
common.ROOT.mkdir(exist_ok=True)
os.environ['DATA_DIR'] = str(common.ROOT)
for gpu, mode in enumerate(common.MODES):
    source = actual / 'preflight-training/exp/selective-copying' / f'benchmark-gpu{gpu}-{mode}'
    for folder in ['model_best', 'model']:
        shutil.copytree(source / folder, common.run_dir(mode) / folder)
    shutil.copy2(source / 'run_config.json', common.run_dir(mode) / 'run_config.json')

import evaluate

evaluate.ROOT = common.ROOT
horizons = [3, 16, 65]
evaluate.ev.build_t_grid = lambda exponent: horizons
task = common.bind_selective_task()
expected = {}
generator = torch.Generator().manual_seed(12345)
for T in horizons:
    inputs, labels = task.make_batch(T, 256, generator=generator)
    positions = ((inputs >= 1) & (inputs <= 8)).nonzero()[:, 1].reshape(256, 10)
    expected[str(T)] = dict(target=labels[:, -10:].tolist(), positions=positions.tolist())

for mode in common.MODES:
    for checkpoint in ['best', 'final']:
        sys.argv = ['evaluate.py', '--mode', mode, '--checkpoint', checkpoint]
        evaluate.main()
        records = json.loads((common.ROOT / mode / f'digits_{checkpoint}.json').read_text())
        payload = json.loads((common.ROOT / mode / f'results_{checkpoint}.json').read_text())
        assert payload['task'] == 'selective-copying'
        assert payload['task_sha256'] == common.sha(common.TASK_SOURCE)
        assert list(map(int, records)) == horizons
        for T in horizons:
            batches = records[str(T)]
            assert len(batches) == 1
            batch = batches[0]
            assert {k: batch[k] for k in ['target', 'positions']} == expected[str(T)]
            assert batch['memory'] == batch['target']
            target, prediction = batch['target'], batch['prediction']
            assert len(target) == len(prediction) == 256
            metric = payload['results'][str(T)]
            assert sum(t == p for t, p in zip(target, prediction)) / 256 == metric['string_acc']
            assert sum(a == b for t, p in zip(target, prediction) for a, b in zip(t, p)) / 2560 == metric['token_acc']

common.save(common.HERE / 'evaluation_smoke.json', dict(
    passed=True, task='selective-copying', modes=list(common.MODES),
    checkpoints=['best', 'final'], standard_horizons=horizons, standard_samples=256,
    checkpoint_training_steps=300, exact_sample_stream_replayed=True,
    target_positions_and_predictions_audited=True))
