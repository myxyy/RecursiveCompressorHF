"""Four real-checkpoint smoke evaluations with independently replayed task data."""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import torch
from common import HERE, ROOT, SOURCE, MODES, GPUS, name, save

def main():
    smoke = ROOT / 'evaluation-smoke'; smoke.mkdir()
    torch.set_num_threads(1)
    for gpu, mode in zip(GPUS, MODES):
        src = ROOT / 'preflight-training/exp' / mode / f'benchmark-gpu{gpu}-{mode}'
        dest = smoke / 'exp' / mode / name(mode)
        for folder in ['model_best', 'model']: shutil.copytree(src / folder, dest / folder)
        shutil.copy2(src / 'run_config.json', dest / 'run_config.json')
        spec = importlib.util.spec_from_file_location('reference_' + mode, SOURCE / 'exp' / mode / 'task.py')
        task = importlib.util.module_from_spec(spec); spec.loader.exec_module(task)
        gen = torch.Generator().manual_seed(12345); expected = {}
        for T in [3, 16, 65]:
            ids, labels = task.make_batch(T, 256, generator=gen)
            positions = ((ids >= 1) & (ids <= 8)).nonzero()[:, 1].reshape(256, 10)
            expected[str(T)] = dict(target=labels[:, -10:].tolist(), positions=positions.tolist())
        for cp in ['best', 'final']:
            code = ('import common; from pathlib import Path; import sys; '
                    f'common.ROOT=Path({str(smoke)!r}); import evaluate; '
                    f'sys.argv=["evaluate.py","--mode",{mode!r},"--checkpoint",{cp!r},"--smoke"]; evaluate.main()')
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(GPUS[0]), 'LOGKV_TASK': mode,
                   'DATA_DIR': str(smoke), 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
            subprocess.run([sys.executable, '-c', code], cwd=HERE, env=env, check=True, timeout=180)
            records = json.loads((smoke / mode / f'digits_{cp}.json').read_text())
            payload = json.loads((smoke / mode / f'results_{cp}.json').read_text())
            assert payload['task'] == mode
            for T in [3, 16, 65]:
                batches = records[str(T)]; assert len(batches) == 1
                b = batches[0]; assert {k: b[k] for k in ['target', 'positions']} == expected[str(T)]
                assert b['target'] == b['memory']
                correct = sum(a == v for t, p in zip(b['target'], b['prediction']) for a, v in zip(t, p))
                exact = sum(t == p for t, p in zip(b['target'], b['prediction']))
                assert correct / 2560 == payload['results'][str(T)]['token_acc']
                assert exact / 256 == payload['results'][str(T)]['string_acc']
    save(HERE / 'evaluation_smoke.json', dict(passed=True, tasks=list(MODES),
        checkpoints=['best', 'final'], horizons=[3, 16, 65], samples=256,
        benchmark_training_steps=300, targets_and_positions_replayed=True))

if __name__ == '__main__': main()
