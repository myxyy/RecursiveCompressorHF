"""Two GPUs, fixed M=10, one bounded train/evaluation stage; no long-probe chaining."""
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/mnt/raid0/RecursiveCompressor/experiments/logkv-fixed10-combined-20260908')
SOURCE = Path('/mnt/raid0/RecursiveCompressor/experiments/logkv-position-study-20260907/source')
NAME = 'combined-no-decay-fixed10-20260908'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'logs').mkdir(exist_ok=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip()
    assert commit == '24b360cf85712fde3ee7a2da4d61e2eb45350a51'
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=SOURCE, text=True)
    for task in ('copying', 'selective-copying'):
        if (ROOT / f'{task}.json').exists() or (ROOT / 'exp' / task / NAME).exists():
            raise FileExistsError(task)
    deadline = time.monotonic() + 8 * 3600

    def worker(pair):
        gpu, task = pair
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '1',
               'MKL_NUM_THREADS': '1', 'DATA_DIR': str(ROOT),
               'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}
        run_dir = ROOT / 'exp' / task / NAME
        manifest = dict(task=task, gpu=gpu, commit=commit, source=str(SOURCE),
                        run_dir=str(run_dir), started=now(), state='starting', commands=[])
        path = ROOT / f'{task}.json'
        commands = [('train', [sys.executable, f'exp/{task}/train.py', '--run-name', NAME,
                     '--arch', 'logkv', '--compressor-position-transform', '--relative-position-kv',
                     '--level-decay-scale', '0', '--gated-attention', '--self-slot',
                     '--phase-levels', '2', '--t-dist', 'loguniform', '--max-t', '2028',
                     '--steps', '50000', '--batch-size', '64', '--grad-accum', '1',
                     '--lr', '0.0003', '--warmup', '1000', '--d-model', '512',
                     '--num-heads', '8', '--d-ff', '1024', '--num-layers', '2',
                     '--chunk-size', '4', '--loss-positions', 'all', '--seed', '0', '--device', '0'])]
        for checkpoint in ('best', 'final'):
            commands.append((checkpoint, [sys.executable, f'exp/{task}/evaluate.py',
                            '--run-name', NAME, '--samples', '256', '--max-t-exp', '17',
                            '--seed', '12345', '--precision', 'bf16',
                            '--checkpoint', checkpoint, '--device', '0']))
        try:
            for stage, command in commands:
                entry = dict(stage=stage, command=command, started=now(), returncode=None)
                manifest['state'] = stage
                manifest['commands'].append(entry)
                save(path, manifest)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Eight-hour stage limit; confirmation required to continue')
                with (ROOT / 'logs' / f'{task}-{stage}.log').open('w') as log:
                    result = subprocess.run(command, cwd=SOURCE, env=env, stdout=log,
                                            stderr=subprocess.STDOUT, timeout=remaining)
                entry.update(returncode=result.returncode, finished=now())
                save(path, manifest)
                if result.returncode:
                    raise RuntimeError(f'{stage}: exit {result.returncode}')
                if stage != 'train':
                    for filename in ('results.json', 'plot.png'):
                        p = Path(filename)
                        shutil.copy2(run_dir / p, run_dir / f'{p.stem}_{stage}{p.suffix}')
            manifest['state'] = 'complete-awaiting-review'
        except Exception as exc:
            manifest.update(state='stopped', error=str(exc))
            raise
        finally:
            manifest['finished'] = now()
            save(path, manifest)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(worker, enumerate(('copying', 'selective-copying'))))


if __name__ == '__main__':
    main()
