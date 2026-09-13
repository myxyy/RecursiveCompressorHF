"""CPU paired prediction counts on identical two/three-layer evaluation data."""
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
            exact_two=sum(map(all, c2)), exact_three=sum(map(all, c3)),
            exact_rescued=sum(not all(a) and all(b) for a, b in zip(c2, c3)),
            exact_regressed=sum(all(a) and not all(b) for a, b in zip(c2, c3)),
            digit_errors_two=[sum(not c[i] for c in c2) for i in range(10)],
            digit_errors_three=[sum(not c[i] for c in c3) for i in range(10)],
            digit_rescued=[sum(not a[i] and b[i] for a, b in zip(c2, c3)) for i in range(10)],
            digit_regressed=[sum(a[i] and not b[i] for a, b in zip(c2, c3)) for i in range(10)]))
    return rows


def main():
    dest = HERE / 'results'
    baseline = HERE.parent / 'logkv-no-position-main-20260913/results'
    status = BASE_ROOT / 'campaign.json'
    # Baseline supervision is independent. Never retry or change its artifacts.
    state = json.loads(status.read_text())['state'] if status.exists() else 'missing'
    if state != 'complete-awaiting-review' or not (baseline / 'review.json').exists():
        save(dest / 'depth_comparison.json', dict(state='pending-baseline', baseline_state=state))
        return
    assert json.loads((baseline / 'review.json').read_text())['passed']
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
    save(dest / 'depth_comparison.json', dict(state='complete', passed=True,
        identical_memories_and_positions=True, baseline_review_sha256=sha(baseline / 'review.json'),
        input_hashes=inputs, rows=rows,
        limitations='One seed; three layers also add parameters. Historical overlapping-layout degeneracy is not a proof for the current layout.'))


if __name__ == '__main__':
    main()
