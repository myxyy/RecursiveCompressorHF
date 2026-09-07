"""Regenerate tables and figures from this archive and the prior position-bias archive."""
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent/'logkv-relative-20260907'
TASKS = ['copying', 'selective-copying']
MODES = ['phase2', 'retrieval', 'half', 'base', 'learned']
NEW = ['half', 'base', 'learned']
COLORS = dict(phase2='#0072B2', retrieval='#888888', half='#009E73', base='#D55E00', learned='#AA4499')
GRID = set(range(1, 15)) | {2**k for k in range(4, 18)} | {3*2**(k-1) for k in range(4, 17)}
EXPECTED = {'half': math.log(4)/6, 'base': math.log(4)/3, 'learned': math.log(4)/3}
results, records, summary, rows = {}, {}, [], []
common = None
excluded = {'run_name', 'phase_emb', 'relative_position_bias', 'num_params',
            'compressor_decay', 'learnable_compressor_decay'}
for task in TASKS:
    for mode in MODES:
        root = ROOT if mode in NEW else PRIOR
        original = 'relative' if mode == 'retrieval' else mode
        manifest = json.loads((root/f'{original}-{task}.json').read_text())
        assert manifest['state'] == 'complete'
        assert len(manifest['commands']) == 3 and all(c['returncode'] == 0 for c in manifest['commands'])
        run = root/'runs'/f'{original}-{task}'
        cfg = json.loads((run/'run_config.json').read_text())
        assert cfg['phase_emb'] == (mode == 'phase2')
        assert cfg['relative_position_bias'] == (mode != 'phase2')
        comparable = {k:v for k,v in cfg.items() if k not in excluded}
        if common is None:
            common = comparable
        assert comparable == common
        if mode in NEW:
            assert cfg['compressor_decay'] == EXPECTED[mode]
            assert cfg['learnable_compressor_decay'] == (mode == 'learned')
        data = [json.loads(line) for line in (run/'train_log.jsonl').read_text().splitlines()]
        train = [r for r in data if 'loss' in r]
        assert [r['step'] for r in train] == list(range(100, 50001, 100))
        assert all(math.isfinite(r[k]) for r in train for k in ['loss', 'ema_loss'])
        records[task, mode] = train
        best = json.loads((run/'best.json').read_text())
        entry = dict(task=task, mode=mode, source_commit=manifest['commit'], gpu=manifest['gpu'],
                     num_params=cfg['num_params'], best=best, final_train=train[-1],
                     first_interval_string_099=next((r['step'] for r in train if r['string_acc'] >= .99), None))
        if mode == 'learned':
            assert all(len(r['compressor_slopes']) == 2 and all(len(a) == 8 for a in r['compressor_slopes']) for r in train)
            assert all(math.isfinite(a) and a > 0 for r in train for layer in r['compressor_slopes'] for a in layer)
        for checkpoint in ['best', 'final']:
            model_cfg = json.loads((run/f'model_config_{checkpoint}.json').read_text())
            for k in ['phase_emb', 'relative_position_bias', 'phase_levels', 'self_slot', 'gated_attention', 'learnable_decay', 'chunk_size']:
                assert model_cfg[k] == cfg[k]
            if mode in NEW:
                assert model_cfg['compressor_decay'] == cfg['compressor_decay']
                assert model_cfg['learnable_compressor_decay'] == cfg['learnable_compressor_decay']
                slopes = json.loads((run/f'compressor_slopes_{checkpoint}.json').read_text())
                entry[f'slopes_{checkpoint}'] = slopes
                if mode == 'learned':
                    rec = next(r for r in train if r['step'] == (best['step'] if checkpoint == 'best' else 50000))
                    assert all(math.isclose(a,b,rel_tol=1e-6,abs_tol=1e-7)
                               for x,y in zip(slopes,rec['compressor_slopes']) for a,b in zip(x,y))
            d = json.loads((run/f'results_{checkpoint}.json').read_text())
            assert d['samples'] == 256 and d['train_max_t'] == 2028 and d['precision'] == 'bf16'
            assert set(map(int,d['results'])) == GRID
            results[task, mode, checkpoint] = d['results']
            for t, score in d['results'].items():
                assert score['n'] == 256 and 0 <= score['string_acc'] <= score['token_acc'] <= 1
                rows.append(dict(task=task, mode=mode, checkpoint=checkpoint, T=int(t), **score))
        summary.append(entry)
(ROOT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
with (ROOT/'comparison.csv').open('w') as f:
    w = csv.DictWriter(f, fieldnames=['task','mode','checkpoint','T','token_acc','string_acc','n'], lineterminator='\n')
    w.writeheader(); w.writerows(rows)

table = ['# Representative horizons', '', 'Each cell is token accuracy / string accuracy (%).',
         'Training T <= 2028; evaluation n=256 per horizon. phase2/retrieval are prior-run baselines.', '']
for task in TASKS:
    for checkpoint in ['best','final']:
        table += [f'## {task}: {checkpoint}', '', '| T | ' + ' | '.join(MODES) + ' |', '|---:|'+'---:|'*len(MODES)]
        for t in [1,16,64,256,1024,2048,8192,32768,131072]:
            cells = []
            for mode in MODES:
                d = results[task,mode,checkpoint][str(t)]
                cells.append(f"{100*d['token_acc']:.2f} / {100*d['string_acc']:.2f}")
            table.append(f'| {t:,} | '+' | '.join(cells)+' |')
        table.append('')
(ROOT/'tables.md').write_text('\n'.join(table))

for checkpoint in ['best','final']:
    fig, axes = plt.subplots(2,2,figsize=(13,8),sharex=True,sharey=True)
    for r, task in enumerate(TASKS):
        for c, metric in enumerate(['token_acc','string_acc']):
            ax=axes[r,c]
            for mode in MODES:
                d=results[task,mode,checkpoint]; ts=sorted(map(int,d))
                ax.plot(ts,[d[str(t)][metric] for t in ts],color=COLORS[mode],label=mode,lw=1.6)
            ax.axvline(2028,color='#BBBBBB',ls=':',label='training max T')
            if c==0: ax.axhline(1/8,color='#CCCCCC',ls=':')
            ax.set_xscale('log'); ax.set_ylim(-.02,1.02); ax.grid(alpha=.25)
            ax.set_title(f'{task}: {metric.replace("_"," ")}')
            ax.set_xlabel('Memory horizon T'); ax.set_ylabel('Accuracy')
    axes[0,0].legend(fontsize=8,ncol=2)
    fig.suptitle(f'LogKV compressor position decay: {checkpoint} checkpoints\n50k steps, train seed 0; evaluation seed 12345, n=256 per horizon')
    fig.tight_layout(); fig.savefig(ROOT/f'comparison_{checkpoint}.png',dpi=180); plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
for r,task in enumerate(TASKS):
    for mode in MODES:
        train=records[task,mode]; steps=[v['step'] for v in train]
        style='--' if mode not in NEW else '-'
        axes[r,0].plot(steps,[max(1e-12,v['ema_loss']) for v in train],style,color=COLORS[mode],label=mode)
        axes[r,1].plot(steps,[v['string_acc'] for v in train],style,color=COLORS[mode],label=mode,alpha=.8)
    axes[r,0].set_yscale('log'); axes[r,0].set_ylabel('EMA cross entropy')
    axes[r,1].set_ylabel('Training interval string accuracy'); axes[r,1].set_ylim(-.02,1.02)
    for ax in axes[r]:
        ax.set_title(task); ax.set_xlabel('Step'); ax.grid(alpha=.25); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(ROOT/'learning.png',dpi=180); plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
for r,task in enumerate(TASKS):
    train=records[task,'learned']
    for layer in range(2):
        ax=axes[r,layer]
        for h in range(8):
            ax.plot([v['step'] for v in train],[v['compressor_slopes'][layer][h] for v in train],label=f'head {h}')
        ax.set_title(f'{task}: layer {layer+1}'); ax.set_ylabel('Compressor slope alpha')
        ax.set_xlabel('Step'); ax.grid(alpha=.25); ax.legend(fontsize=8,ncol=4)
fig.tight_layout(); fig.savefig(ROOT/'slopes.png',dpi=180); plt.close(fig)
print(f'Validated {len(rows)} evaluation rows (new runs plus prior baselines) and generated figures/tables.')
