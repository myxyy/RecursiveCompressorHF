"""CPU paired prediction counts for fixed decay versus fixed zero bias on identical data."""
import json
from common import BASE_ROOT, HERE, MODES, save, sha


def compare_records(two, three):
    assert two.keys() == three.keys()
    rows = []
    for key in sorted(two, key=int):
        def flatten(records, field):
            return [x for batch in records[key] for x in batch[field]]
        for field in ['target', 'memory', 'positions']:
            assert flatten(two, field) == flatten(three, field), (key, field)
        target = flatten(two, 'target')
        p2, p3 = flatten(two, 'prediction'), flatten(three, 'prediction')
        assert len(target) == len(p2) == len(p3) == 256
        assert all(len(v) == 10 for v in target + p2 + p3)
        c2 = [[a == b for a, b in zip(t, p)] for t, p in zip(target, p2)]
        c3 = [[a == b for a, b in zip(t, p)] for t, p in zip(target, p3)]
        rows.append(dict(T=int(key), n=len(target),
            exact_baseline=sum(map(all, c2)), exact_zero_bias=sum(map(all, c3)),
            exact_rescued=sum(not all(a) and all(b) for a, b in zip(c2, c3)),
            exact_regressed=sum(all(a) and not all(b) for a, b in zip(c2, c3)),
            digit_errors_baseline=[sum(not c[i] for c in c2) for i in range(10)],
            digit_errors_zero_bias=[sum(not c[i] for c in c3) for i in range(10)],
            digit_rescued=[sum(not a[i] and b[i] for a, b in zip(c2, c3)) for i in range(10)],
            digit_regressed=[sum(a[i] and not b[i] for a, b in zip(c2, c3)) for i in range(10)]))
    return rows


def main():
    dest = HERE / 'results'
    baseline = HERE.parent / 'logkv-causal-conv-20260914/results'
    status = BASE_ROOT / 'campaign.json'
    # Baseline supervision is independent. Never retry or change its artifacts.
    state = json.loads(status.read_text())['state'] if status.exists() else 'missing'
    if state != 'complete-awaiting-review' or not (baseline / 'review.json').exists():
        save(dest / 'baseline_comparison.json', dict(state='pending-baseline', baseline_state=state))
        return
    review = json.loads((baseline / 'review.json').read_text())
    assert review['passed']
    for name, expected in review['result_hashes'].items():
        assert sha(baseline / name) == expected
    rows, inputs = [], {}
    for mode in MODES:
        for cp in ['best', 'final']:
            old = baseline / mode / f'digits_{cp}.json'
            new = dest / mode / f'digits_{cp}.json'
            for path in [old, new]:
                inputs[str(path)] = sha(path)
            paired = compare_records(json.loads(old.read_text()), json.loads(new.read_text()))
            assert len(paired) == 41
            rows.extend(dict(task=mode, checkpoint=cp, **r) for r in paired)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for axrow, mode in zip(axes, MODES):
        for ax, cp in zip(axrow, ['best', 'final']):
            rr = [r for r in rows if r['task'] == mode and r['checkpoint'] == cp]
            for field, label in [('exact_baseline', 'Fixed decay (-i log C)'), ('exact_zero_bias', 'Fixed zero bias')]:
                ax.plot([r['T'] for r in rr], [r[field]/r['n'] for r in rr], '.-', label=label)
            ax.axvline(2028, color='gray', linestyle='--', linewidth=1)
            ax.set(xscale='log', ylim=(-.02, 1.02), xlabel='T', ylabel='Exact string accuracy', title=f'{mode}: {cp}')
            ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(dest/'fixed_decay_vs_zero_bias.png', dpi=160); plt.close(fig)
    save(dest / 'baseline_comparison.json', dict(state='complete', passed=True,
        identical_memories_and_positions=True, baseline_review_sha256=sha(baseline / 'review.json'),
        input_hashes=inputs, rows=rows,
        limitations='One seed; same parameter count and identical initial weights. Only the fixed level bias is removed; current training source is archived separately.'))


if __name__ == '__main__':
    main()
