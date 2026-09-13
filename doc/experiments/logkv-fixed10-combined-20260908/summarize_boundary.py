"""Validate saved per-sample boundary predictions and plot the approved sweep."""
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('/mnt/raid0/RecursiveCompressor/experiments/logkv-fixed10-combined-20260908')


def main():
    data = json.loads((ROOT / 'boundary/results.json').read_text())
    answers = json.loads((ROOT / 'boundary/answers.json').read_text())
    assert data['complete'] and data['samples'] == 256
    assert [r['T'] for r in data['results']] == list(range(4064,4113))
    original = json.loads((HERE / 'summary.json').read_text())
    assert data['checkpoint_sha256'] == original['copying']['checkpoint_sha256']['final']['model.safetensors']
    targets = np.array([list(map(int,s)) for s in answers['targets']])
    assert targets.shape == (256,10) and ((targets >= 1) & (targets <= 8)).all()
    before_correct = before_count = after_correct = after_count = 0
    for r in data['results']:
        predictions = np.array([list(map(int,s)) for s in answers['predictions'][str(r['T'])]])
        assert predictions.shape == targets.shape
        correct = predictions == targets
        assert r['n'] == 256 and r['token_correct'] == int(correct.sum())
        assert r['string_correct'] == int(correct.all(1).sum())
        assert r['digit_correct'] == correct.sum(0).tolist()
        assert r['token_acc'] == r['token_correct']/2560 and r['string_acc'] == r['string_correct']/256
        assert r['digit_acc'] == (correct.mean(0)).tolist()
        before = r['T'] + 10 + np.arange(10) < 4096
        before_correct += int(correct[:,before].sum()); before_count += 256 * int(before.sum())
        after_correct += int(correct[:,~before].sum()); after_count += 256 * int((~before).sum())
    out = HERE / 'boundary'
    out.mkdir(exist_ok=True)
    for name in ('results.json','answers.json'):
        shutil.copy2(ROOT / 'boundary' / name, out / name)
    shutil.copy2(ROOT / 'boundary.log', out / 'run.log')
    rows = data['results']
    with (out / 'metrics.csv').open('w') as f:
        writer = csv.writer(f,lineterminator='\n')
        writer.writerow(['T','token_correct','string_correct','n']+[f'digit_{i}_correct' for i in range(1,11)])
        writer.writerows([[r['T'],r['token_correct'],r['string_correct'],256]+r['digit_correct'] for r in rows])
    summary = dict(first_nonperfect_T=next(r['T'] for r in rows if r['string_correct']<256),
                   perfect_T=[r['T'] for r in rows if r['string_correct']==256],
                   before_boundary=dict(correct=before_correct,total=before_count),
                   from_boundary=dict(correct=after_correct,total=after_count),
                   elapsed_sec=data['elapsed_sec'])
    control = json.loads((ROOT / 'boundary/batch_check.json').read_text())
    assert control['complete']
    assert [(r['T'],r['batch']) for r in control['results']] == [(4076,127),(4077,128)]
    for r in control['results']:
        predictions = np.array([list(map(int,s)) for s in r['predictions']])
        correct = predictions == targets
        assert r['token_correct'] == int(correct.sum())
        assert r['string_correct'] == int(correct.all(1).sum())
        assert r['digit_correct'] == correct.sum(0).tolist()
        assert r['identical_predictions'] and r['predictions'] == answers['predictions'][str(r['T'])]
    shutil.copy2(ROOT / 'boundary/batch_check.json',out / 'batch_check.json')
    shutil.copy2(ROOT / 'boundary-batch-check.log',out / 'batch_check.log')
    summary['batch_size_control_identical_predictions'] = True
    summary['batch_size_control_elapsed_sec'] = control['elapsed_sec']
    (out / 'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,ax = plt.subplots(2,1,figsize=(11,8),sharex=True,layout='constrained')
    ts = [r['T'] for r in rows]
    ax[0].plot(ts,[100*r['string_acc'] for r in rows],'o-',label='Exact string accuracy',ms=3)
    ax[0].plot(ts,[100*r['token_acc'] for r in rows],'s-',label='Token accuracy',ms=3)
    for t,label in ((4077,'Last answer reaches position 4096'),(4086,'First answer reaches position 4096')):
        ax[0].axvline(t,color='gray',ls=':',lw=1,label=label)
    ax[0].set(ylabel='Accuracy (%)',ylim=(-2,102),title='Fixed-M10 Copying: paired 256 memories per T')
    ax[0].legend(fontsize=8);ax[0].grid(alpha=.2)
    heat = np.array([r['digit_acc'] for r in rows]).T * 100
    plot = ax[1].pcolormesh(np.arange(4063.5,4113),np.arange(.5,11),heat,vmin=0,vmax=100,cmap='viridis')
    ax[1].plot(4087-np.arange(1,11),np.arange(1,11),'w--',lw=2,label='Answer position = 4096')
    ax[1].set(xlabel='T',ylabel='Answer digit (1-based)',yticks=range(1,11),xlim=(4063.5,4112.5))
    ax[1].invert_yaxis();ax[1].legend(fontsize=8,loc='upper right')
    fig.colorbar(plot,ax=ax[1],label='Per-digit accuracy (%)')
    fig.savefig(out / 'boundary.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
