"""Validate and archive completed fixed-M10 results, then compare with historical phase2."""
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = Path('/mnt/raid0/RecursiveCompressor/experiments/logkv-fixed10-combined-20260908')
OLD = HERE.parent / 'logkv-refine-20260906' / 'runs'
NAME = 'combined-no-decay-fixed10-20260908'
TASKS = ('copying', 'selective-copying')
GRID = sorted(set(range(1, 15)) | {2**k for k in range(4, 18)} |
              {3 * 2**(k-1) for k in range(4, 17)})


def read(path):
    return json.loads(path.read_text())


def main():
    for task in TASKS:
        assert read(ROOT / f'{task}.json')['state'] == 'complete-awaiting-review'
    shutil.copytree(ROOT / 'logs', HERE / 'logs', dirs_exist_ok=True)
    (HERE / 'gpu_environment.txt').write_text('\n'.join(
        line.rstrip() for line in (ROOT / 'gpu_environment.txt').read_text().splitlines()) + '\n')
    shutil.copy2(ROOT / 'environment.json', HERE / 'environment.json')
    failed = ROOT / 'attempt1-allocator-oom'
    if failed.exists():
        shutil.copytree(failed / 'logs', HERE / 'attempt1-allocator-oom' / 'logs', dirs_exist_ok=True)
        for name in ('copying.json', 'selective-copying.json', 'run.py'):
            shutil.copy2(failed / name, HERE / 'attempt1-allocator-oom' / name)
    rows, summary = [], {}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    learning, lax = plt.subplots(2, 2, figsize=(12, 8))
    for ti, task in enumerate(TASKS):
        manifest = read(ROOT / f'{task}.json')
        assert manifest['state'] == 'complete-awaiting-review', manifest
        assert len(manifest['commands']) == 3
        assert all(c['returncode'] == 0 for c in manifest['commands'])
        shutil.copy2(ROOT / f'{task}.json', HERE / f'{task}.json')
        run = ROOT / 'exp' / task / NAME
        dest = HERE / 'runs' / task
        dest.mkdir(parents=True, exist_ok=True)
        cfg = read(run / 'run_config.json')
        assert cfg['steps'] == 50000 and cfg['batch_size'] == 64 and cfg['grad_accum'] == 1
        assert cfg['compressor_position_transform'] and cfg['relative_position_kv']
        assert cfg['level_decay_scale'] == 0 and not cfg['phase_emb']
        assert not cfg['relative_position_bias'] and cfg['compressor_decay'] == 0
        logs = [json.loads(s) for s in (run / 'train_log.jsonl').read_text().splitlines()]
        train = [r for r in logs if 'elapsed_sec' in r]
        quick = [r for r in logs if 'quick_eval' in r]
        assert [r['step'] for r in train] == list(range(100, 50001, 100))
        assert [r['step'] for r in quick] == list(range(5000, 50001, 5000))
        assert all(math.isfinite(r['loss']) and math.isfinite(r['ema_loss']) for r in train)
        best = read(run / 'best.json')
        expected = max(train, key=lambda r: (r['string_acc'], r['token_acc'], -r['ema_loss']))
        assert best['step'] == expected['step']
        summary[task] = dict(best_step=best['step'], train_hours=train[-1]['elapsed_sec']/3600,
                             final_train=train[-1], quick_eval_history=quick, checkpoints={})
        for filename in ('run_config.json', 'train_log.jsonl', 'best.json',
                         'results_best.json', 'results_final.json'):
            shutil.copy2(run / filename, dest / filename)
        for cp, folder in (('best', 'model_best'), ('final', 'model')):
            shutil.copy2(run / folder / 'config.json', dest / f'model_config_{cp}.json')
        hashes = {cp: {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted((run / folder).glob('*.safetensors'))}
                  for cp, folder in (('best','model_best'), ('final','model'))}
        assert all(hashes.values())
        summary[task]['checkpoint_sha256'] = hashes
        summary[task]['identical_checkpoints'] = hashes['best'] == hashes['final']
        if hashes['best'] == hashes['final']:
            assert read(run / 'results_best.json')['results'] == read(run / 'results_final.json')['results']
        for mode, folder in (('phase2 (historical)', OLD / f'refined-{task}'),
                             ('combined-no-decay', run)):
            tr = [json.loads(s) for s in (folder / 'train_log.jsonl').read_text().splitlines()]
            tr = [r for r in tr if 'elapsed_sec' in r]
            lax[ti, 0].plot([r['step'] for r in tr], [max(r['ema_loss'], 1e-12) for r in tr], label=mode)
            lax[ti, 1].plot([r['step'] for r in tr], [r['string_acc']*100 for r in tr], label=mode, alpha=.8)
            for cp in ('best', 'final'):
                data = read(folder / f'results_{cp}.json')
                assert data['samples'] == 256 and data['precision'] == 'bf16'
                assert sorted(map(int, data['results'])) == GRID
                for t in GRID:
                    cell = data['results'][str(t)]
                    assert cell['n'] == 256
                    for metric, denom in (('token_acc', 2560), ('string_acc', 256)):
                        assert 0 <= cell[metric] <= 1
                        assert abs(cell[metric]*denom-round(cell[metric]*denom)) < 1e-8
                    rows.append(dict(task=task, mode=mode, checkpoint=cp, T=t, **cell))
                label = f'{mode}, {cp}'
                for ai, metric in enumerate(('token_acc', 'string_acc')):
                    axes[ti, ai].plot(GRID, [data['results'][str(t)][metric]*100 for t in GRID],
                                      label=label, linestyle='-' if cp == 'best' else '--')
                if mode == 'combined-no-decay':
                    exact = [t for t in GRID if data['results'][str(t)]['string_acc'] == 1]
                    contiguous = []
                    for t in GRID:
                        if t not in exact:
                            break
                        contiguous.append(t)
                    summary[task]['checkpoints'][cp] = dict(
                        perfect_cells=len(exact), largest_perfect_T=max(exact, default=None),
                        perfect_prefix_through_T=max(contiguous, default=None),
                        selected={str(t): data['results'][str(t)] for t in (1,16,64,256,1024,2048,16384,131072)})
        for ai, metric in enumerate(('Token accuracy (%)', 'Exact string accuracy (%)')):
            axes[ti, ai].set(title=f'{task}: {metric}', xlabel='T', xscale='log', ylim=(-2, 102))
            axes[ti, ai].axvline(2028, color='gray', linestyle=':', linewidth=1)
            axes[ti, ai].grid(alpha=.2)
            axes[ti, ai].legend(fontsize=7)
        lax[ti, 0].set(title=f'{task}: EMA loss', xlabel='Step', yscale='log')
        lax[ti, 1].set(title=f'{task}: training string accuracy (%)', xlabel='Step', ylim=(-2,102))
        for ax in lax[ti]:
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    assert len(rows) == 328
    with (HERE / 'comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    (HERE / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    for f, name in ((fig, 'comparison.png'), (learning, 'learning.png')):
        f.tight_layout(); f.savefig(HERE / name, dpi=160); plt.close(f)
    table = ['| Task | checkpoint | T | token % | string % | old phase2 string % |',
             '|---|---|---:|---:|---:|---:|']
    for task in TASKS:
        for cp in ('best','final'):
            for t in (1,16,64,256,1024,2048,16384,131072):
                r = next(r for r in rows if (r['task'],r['checkpoint'],r['T'],r['mode']) ==
                         (task,cp,t,'combined-no-decay'))
                old = next(r for r in rows if (r['task'],r['checkpoint'],r['T'],r['mode']) ==
                           (task,cp,t,'phase2 (historical)'))
                table.append(f"| {task} | {cp} | {t:,} | {100*r['token_acc']:.2f} | "
                             f"{100*r['string_acc']:.2f} | {100*old['string_acc']:.2f} |")
    (HERE / 'tables.md').write_text('\n'.join(table)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
