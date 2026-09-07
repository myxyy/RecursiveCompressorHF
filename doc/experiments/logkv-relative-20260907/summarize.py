"""Regenerate all figures/tables from the archived run files (no GPU needed)."""
import csv
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
MODES = ['phase2', 'none', 'relative']
COLORS = {'phase2': '#0072B2', 'none': '#888888', 'relative': '#D55E00'}
TASKS = ['copying', 'selective-copying']
manifests = {(m['task'], m['mode']): m for p in sorted(ROOT.glob('*-copying.json'))
             for m in [json.loads(p.read_text())]}
assert len(manifests) == 6
grid = set(range(1, 15)) | {2**k for k in range(4, 18)} | {3*2**(k-1) for k in range(4, 17)}
results, summary, rows = {}, [], []
common_config = None
for task in TASKS:
    for mode in MODES:
        m = manifests[task, mode]
        if m['state'] != 'complete':
            continue
        assert m['state'] == 'complete' and all(c['returncode'] == 0 for c in m['commands'])
        run = ROOT / 'runs' / f'{mode}-{task}'
        records = [json.loads(line) for line in (run/'train_log.jsonl').read_text().splitlines()]
        train = [r for r in records if 'loss' in r]
        assert train[-1]['step'] == 50000
        assert all(math.isfinite(r[k]) for r in train for k in ['loss', 'ema_loss'])
        best = json.loads((run/'best.json').read_text())
        cfg = json.loads((run/'run_config.json').read_text())
        assert cfg['phase_emb'] == (mode == 'phase2')
        assert cfg['relative_position_bias'] == (mode == 'relative')
        common = {k: v for k, v in cfg.items()
                  if k not in {'run_name', 'phase_emb', 'relative_position_bias', 'num_params'}}
        if common_config is None:
            common_config = common
        assert common == common_config
        summary.append(dict(task=task, mode=mode, commit=m['commit'], gpu=m['gpu'],
                            num_params=cfg['num_params'], best=best, final_train=train[-1],
                            first_interval_string_099=next((r['step'] for r in train if r['string_acc'] >= .99), None)))
        for checkpoint in ['best', 'final']:
            model_cfg = json.loads((run/f'model_config_{checkpoint}.json').read_text())
            for key in ['phase_emb', 'relative_position_bias', 'phase_levels', 'self_slot',
                        'gated_attention', 'learnable_decay', 'chunk_size']:
                assert model_cfg[key] == cfg[key]
            d = json.loads((run/f'results_{checkpoint}.json').read_text())
            assert d['samples'] == 256 and d['train_max_t'] == 2028
            assert set(map(int, d['results'])) == grid
            results[task, mode, checkpoint] = d['results']
            for t, scores in d['results'].items():
                assert scores['n'] == 256
                assert all(0 <= scores[k] <= 1 for k in ['token_acc', 'string_acc'])
                assert scores['string_acc'] <= scores['token_acc']
                rows.append(dict(task=task, mode=mode, checkpoint=checkpoint, T=int(t), **scores))
with (ROOT/'comparison.csv').open('w') as f:
    w = csv.DictWriter(f, fieldnames=['task', 'mode', 'checkpoint', 'T', 'token_acc', 'string_acc', 'n'], lineterminator='\n')
    w.writeheader(); w.writerows(rows)
(ROOT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')

table = ['# Representative horizons', '', 'Each cell is token accuracy / string accuracy (%).',
         'Training: T <= 2028. Evaluation: 256 samples per horizon.', '']
for task in TASKS:
    for checkpoint in ['best', 'final']:
        table += [f'## {task}: {checkpoint}', '', '| T | phase2 | none | relative |',
                  '|---:|---:|---:|---:|']
        for t in [1, 16, 64, 256, 1024, 2048, 8192, 32768, 131072]:
            cells = []
            for mode in MODES:
                if (task, mode, checkpoint) not in results:
                    cells.append('pending')
                    continue
                d = results[task, mode, checkpoint][str(t)]
                cells.append(f"{100*d['token_acc']:.2f} / {100*d['string_acc']:.2f}")
            table.append(f'| {t:,} | ' + ' | '.join(cells) + ' |')
        table.append('')
(ROOT/'tables.md').write_text('\n'.join(table))

fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
for r, task in enumerate(TASKS):
    for c, metric in enumerate(['token_acc', 'string_acc']):
        ax = axes[r,c]
        for mode in MODES:
            for checkpoint, style in [('best','-'), ('final','--')]:
                if (task, mode, checkpoint) not in results:
                    continue
                d = results[task, mode, checkpoint]
                ts = sorted(map(int, d))
                ax.plot(ts, [d[str(t)][metric] for t in ts], style, color=COLORS[mode], label=f'{mode} ({checkpoint})', lw=1.6)
        ax.axvline(2028, color='#009E73', ls=':', label='training max T = 2028')
        if c == 0: ax.axhline(1/8, color='#BBBBBB', ls=':', label='chance = 1/8')
        ax.set_xscale('log'); ax.set_ylim(-.02, 1.02); ax.grid(alpha=.25)
        ax.set_title(f'{task}: {metric.replace("_", " ")}')
        ax.set_xlabel('Memory horizon T'); ax.set_ylabel('Accuracy')
axes[0,0].legend(fontsize=8, ncol=2)
axes[1,0].legend(fontsize=8, ncol=2, loc='lower left')
pending = [f'{task}/{mode}' for (task, mode), m in manifests.items() if m['state'] != 'complete']
subtitle = 'Pending: ' + ', '.join(pending) if pending else 'All six runs complete'
fig.suptitle('LogKV position encoding: phase2 vs no encoding vs relative logit bias\n50k steps, train seed 0; evaluation seed 12345, n=256 per horizon\n' + subtitle)
fig.tight_layout(); fig.savefig(ROOT/'comparison.png', dpi=180); plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
for r, task in enumerate(TASKS):
    for mode in MODES:
        run = ROOT/'runs'/f'{mode}-{task}'
        records = [json.loads(line) for line in (run/'train_log.jsonl').read_text().splitlines()]
        records = [r for r in records if 'loss' in r]
        steps = [r['step'] for r in records]
        label = mode + (' (in progress)' if manifests[task, mode]['state'] != 'complete' else '')
        axes[r,0].plot(steps, [max(1e-12,r['ema_loss']) for r in records], color=COLORS[mode], label=label)
        axes[r,1].plot(steps, [r['string_acc'] for r in records], color=COLORS[mode], label=label, alpha=.8)
    axes[r,0].set_yscale('log'); axes[r,0].set_ylabel('EMA cross entropy')
    axes[r,1].set_ylabel('Training interval string accuracy'); axes[r,1].set_ylim(-.02,1.02)
    for ax in axes[r]:
        ax.set_title(task); ax.set_xlabel('Step'); ax.grid(alpha=.25); ax.legend()
fig.tight_layout(); fig.savefig(ROOT/'learning.png', dpi=180); plt.close(fig)
print(f'Validated {len(rows)} evaluation rows and generated comparison/learning plots.')
