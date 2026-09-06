# Regenerate the figures and tables: uv run python doc/experiments/logkv-refine-20260906/summarize.py
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
manifests = {(m['task'], m['layout']): m for p in sorted(ROOT.glob('*-copying.json'))
             for m in [json.loads(p.read_text())]}
colors = {'overlap': '#777777', 'refined': '#0072B2'}
results = {}
rows = []
for (task, layout), m in manifests.items():
    run = ROOT / 'runs' / f"{m['layout']}-{m['task']}"
    for checkpoint in ['best', 'final']:
        path = run / f'results_{checkpoint}.json'
        if not path.exists():
            raise RuntimeError(f'Missing completed evaluation: {path}')
        data = json.loads(path.read_text())
        assert data['samples'] == 256 and data['train_max_t'] == 2028
        results[task, layout, checkpoint] = data['results']
        for t, scores in data['results'].items():
            rows.append({'task': task, 'layout': layout, 'checkpoint': checkpoint, 'T': int(t), **scores})
with (ROOT/'comparison.csv').open('w') as f:
    w = csv.DictWriter(f, fieldnames=['task', 'layout', 'checkpoint', 'T', 'token_acc', 'string_acc', 'n'], lineterminator='\n')
    w.writeheader(); w.writerows(rows)

fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
for row, task in enumerate(['copying', 'selective-copying']):
    for col, metric in enumerate(['token_acc', 'string_acc']):
        ax=axes[row,col]
        for layout in ['overlap', 'refined']:
            for checkpoint, style in [('best','-'), ('final','--')]:
                data=results[task,layout,checkpoint]
                ts=sorted(map(int,data))
                ax.plot(ts,[data[str(t)][metric] for t in ts],style,color=colors[layout],
                        label=f'{layout} ({checkpoint})',lw=1.7)
        ax.axvline(2028,color='#D55E00',ls=':',label='training max T = 2028')
        if metric=='token_acc': ax.axhline(1/8,color='#BBBBBB',ls=':',label='chance = 1/8')
        ax.set_xscale('log'); ax.set_ylim(-.02,1.02)
        ax.set_title(f'{task}: {metric.replace("_", " ")}')
        ax.grid(alpha=.25); ax.set_xlabel('Memory horizon T'); ax.set_ylabel('Accuracy')
axes[0,0].legend(fontsize=8)
fig.suptitle('LogKV overlap removal: matched training, phase2 + gated + self slot\n50k steps, train seed 0; evaluation seed 12345, n=256 per horizon')
fig.tight_layout(); fig.savefig(ROOT/'comparison.png',dpi=180); plt.close(fig)

fig, axes = plt.subplots(2,2,figsize=(13,8),sharex=True)
summary=[]
for row,task in enumerate(['copying','selective-copying']):
    for layout in ['overlap','refined']:
        m=manifests[task,layout]; run=ROOT / 'runs' / f"{m['layout']}-{m['task']}"
        records=[json.loads(line) for line in (run/'train_log.jsonl').read_text().splitlines()]
        records=[r for r in records if 'loss' in r]
        axes[row,0].plot([r['step'] for r in records],[max(1e-12,r['ema_loss']) for r in records],color=colors[layout],label=layout)
        axes[row,1].plot([r['step'] for r in records],[r['string_acc'] for r in records],color=colors[layout],label=layout,alpha=.8)
        best=json.loads((run/'best.json').read_text())
        summary.append({'task':task,'layout':layout,'commit':m['commit'],'best':best,'final_train':records[-1],
                        'first_interval_string_099':next((r['step'] for r in records if r['string_acc']>=.99),None)})
    axes[row,0].set_yscale('log'); axes[row,0].set_ylabel('EMA cross entropy'); axes[row,0].set_title(task)
    axes[row,1].set_ylabel('Training interval string accuracy'); axes[row,1].set_title(task)
    for ax in axes[row]: ax.grid(alpha=.25); ax.set_xlabel('Step'); ax.legend()
fig.tight_layout(); fig.savefig(ROOT/'learning.png',dpi=180); plt.close(fig)
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
