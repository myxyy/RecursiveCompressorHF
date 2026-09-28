"""CPU-only completion recheck and diagnostics from saved predictions/logs."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARCHIVE = Path(__file__).resolve().parent
REPO = ARCHIVE.parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    supervisor = json.loads((args.root / 'supervisor.json').read_text())
    assert supervisor['state'] == 'complete'
    assert json.loads((ARCHIVE / 'review.json').read_text())['passed']
    report = dict(passed=True, checked_at=datetime.now(timezone.utc).isoformat(),
                  gpu_work=False, tasks={}, supervisor=supervisor)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for row, task in enumerate(['copying', 'selective-copying']):
        dest, root = ARCHIVE / task, args.root / task
        manifest = json.loads((root / 'preflight.json').read_text())['source']
        for name, sha in manifest.items():
            assert hashlib.sha256((REPO / name).read_bytes()).hexdigest() == sha, name
        logs = [json.loads(line) for line in (dest / 'train_log.jsonl').read_text().splitlines()]
        summary = dict(workspace_source_matches=True, archives_match_originals=True,
                       source_files=len(manifest), checkpoints={})
        for kind in ['best', 'final']:
            raw = gzip.decompress((dest / f'{kind}.json.gz').read_bytes())
            assert raw == (root / 'results' / f'{kind}.json').read_bytes()
            data = json.loads(raw)
            cells = []
            for cell in data['cells']:
                target, pred = cell['targets'], cell['predictions']
                counts = Counter(map(tuple, pred))
                correct = [sum(a[j] == b[j] for a, b in zip(target, pred)) for j in range(10)]
                assert correct == [256 - n for n in cell['digit_errors']]
                assert sum(correct) == cell['token_correct']
                cells.append(dict(T=cell['T'], unique_predictions=len(counts),
                                  most_common_count=counts.most_common(1)[0][1],
                                  correct_by_digit=correct, first_digit_acc=correct[0]/256,
                                  remaining_nine_acc=sum(correct[1:])/2304))
            summary['checkpoints'][kind] = cells
            last = cells[-1]
            axes[row, 1].plot(range(1, 11), [100*n/256 for n in last['correct_by_digit']],
                              marker='o', label=kind)
        worst = min((r for r in logs if r['step'] >= 10000), key=lambda r:r['string_acc'])
        summary['worst_training_interval_after_10k'] = {k:v for k,v in worst.items() if k != 'mamba_gate'}
        summary['selected_training_intervals'] = [
            {**{k:v for k,v in r.items() if k != 'mamba_gate'},
             'gate_abs_mean':r['mamba_gate']['abs_mean']}
            for r in logs if r['step'] in [16300, 24000, 24100, 24200, 30000, 45600, 45800, 50000]]
        ax = axes[row, 0]
        ax.plot([r['step'] for r in logs], [100*r['string_acc'] for r in logs], color='#0072B2')
        best_step = json.loads((dest / 'best.json').read_text())['step']
        ax.axvline(best_step, linestyle=':', color='black', label=f'best step {best_step:,}')
        ax.set(title=task + ': 100-step training intervals', xlabel='Training step',
               ylabel='Exact match (%)', ylim=(-3, 104))
        ax.legend(fontsize=8)
        axes[row, 1].set(title=task + ': T=131,072, 256 examples', xlabel='Output digit (1-based)',
                          ylabel='Digit accuracy (%)', ylim=(-3, 104), xticks=range(1, 11))
        axes[row, 1].legend()
        report['tasks'][task] = summary
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(ARCHIVE / 'training-and-digits.png', dpi=160)
    plt.close(fig)
    start = datetime.fromisoformat(supervisor['started'])
    end = datetime.fromisoformat(supervisor['finished'])
    report['execution_hours'] = (end-start).total_seconds()/3600
    shutil.copy2(args.root / 'supervisor.json', ARCHIVE / 'supervisor.json')
    (ARCHIVE / 'completion-recheck.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Verified current sources and original prediction archives; saved CPU diagnostics.')


if __name__ == '__main__':
    main()
