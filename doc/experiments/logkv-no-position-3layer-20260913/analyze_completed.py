"""CPU-only review of frozen results; leaves campaign artifacts unchanged."""
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DIRS = {2: HERE.parent / 'logkv-no-position-main-20260913', 3: HERE}
OUT = HERE / 'analysis'
TASKS = ['copying', 'selective-copying']
GRID = sorted(set(range(1, 15)) | {2**k for k in range(4, 18)} |
              {3*2**(k-1) for k in range(4, 17)})


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    OUT.mkdir(exist_ok=True)
    rows, training, checks, records = [], [], [], {}
    for depth, directory in DIRS.items():
        campaign = read(directory / 'campaign.json')
        root = Path(campaign['source_root']).parent if depth == 2 else Path(
            '/mnt/raid0/RecursiveCompressor/experiments/logkv-no-position-3layer-20260913')
        assert campaign == read(root / 'campaign.json')
        assert campaign['state'] == 'complete-awaiting-review'
        assert not campaign['next_stage_queued'] and campaign['whole_gpu_batch_hours'] < 7.5
        for name, expected in campaign['scripts'].items():
            assert sha(directory / name) == expected
        for name, expected in campaign['prerequisite_hashes'].items():
            assert sha(directory / name) == expected
        for name, expected in campaign['source_manifest']['files'].items():
            assert sha(Path(campaign['source_root']) / name) == expected
            assert sha(REPO / name) == expected
        assert sha(root / 'initial_model/model.safetensors') == campaign['initial_weights_sha256']
        review = read(directory / 'results/review.json')
        assert review['passed'] and review['standard_cells'] == 164
        for name, expected in review['result_hashes'].items():
            assert sha(directory / 'results' / name) == expected
        checks.append(dict(depth=depth, passed=True, archived_files=len(review['result_hashes']),
            campaign_sha256=sha(directory / 'campaign.json'), review_sha256=sha(directory / 'results/review.json'),
            finished=campaign['finished'], elapsed_hours=campaign['elapsed_hours'],
            whole_gpu_batch_hours=campaign['whole_gpu_batch_hours']))
        metrics = read(directory / 'results/metrics.json')
        assert len(metrics) == 164
        for task in TASKS:
            local = directory / 'results' / task
            config = read(local / 'run_config.json')
            assert config['num_layers'] == depth and config['steps'] == 50000
            worker = read(local / 'worker.json')
            assert worker['state'] == 'complete'
            assert len(worker['commands']) == 3 and all(c['returncode'] == 0 for c in worker['commands'])
            intervals = [json.loads(s) for s in (local / 'train_log.jsonl').read_text().splitlines()]
            intervals = [r for r in intervals if 'loss' in r]
            assert [r['step'] for r in intervals] == list(range(100, 50001, 100))
            best = read(local / 'best.json')
            assert best['step'] == max(intervals, key=lambda r: (
                r['string_acc'], r['token_acc'], -r['ema_loss']))['step']
            training.append(dict(depth=depth, task=task, best=best, final_interval=intervals[-1],
                last5000_mean={k: float(np.mean([r[k] for r in intervals[-50:]]))
                               for k in ['string_acc', 'token_acc', 'loss']},
                intervals=intervals))
            for cp, folder in [('best', 'model_best'), ('final', 'model')]:
                path = root / 'exp' / task / config['run_name'] / folder / 'model.safetensors'
                assert sha(path) == review['weights'][f'{task}/{cp}'] == worker['weights'][cp]
                data = read(local / f'digits_{cp}.json')
                assert sorted(map(int, data)) == GRID
                for T in GRID:
                    batches = data[str(T)]
                    arrays = {k: np.array([v for b in batches for v in b[k]])
                              for k in ['target', 'prediction', 'memory', 'positions', 'margin']}
                    assert all(v.shape == (256, 10) for v in arrays.values())
                    assert np.array_equal(arrays['target'], arrays['memory'])
                    assert np.isfinite(arrays['margin']).all()
                    records[depth, task, cp, T] = arrays
                    correct = arrays['target'] == arrays['prediction']
                    exact, token = int(correct.all(1).sum()), int(correct.sum())
                    errors = (~correct).sum(0).tolist()
                    metric = next(r for r in metrics if r['mode'] == task and
                                  r['checkpoint'] == cp and r['T'] == T)
                    assert metric['string_correct'] == exact and metric['token_correct'] == token
                    assert metric['digit_errors'] == errors
                    assert metric['string_acc'] == exact / 256 and metric['token_acc'] == token / 2560
                    # Conditional rates prevent equal true adjacent digits from
                    # masquerading as evidence of indistinguishable outputs.
                    different = arrays['target'][:, :-1] != arrays['target'][:, 1:]
                    equal_pred = arrays['prediction'][:, :-1] == arrays['prediction'][:, 1:]
                    adjacent_equal = (different & equal_pred).sum(0)
                    wrong = ~correct
                    rows.append(dict(depth=depth, task=task, checkpoint=cp, T=T,
                        n=256, exact=exact, token_correct=token, digit_errors=errors,
                        prediction_matches_target_position=(arrays['prediction'][:, :, None] ==
                            arrays['target'][:, None, :]).sum(0).tolist(),
                        adjacent_equal_predictions=equal_pred.sum(0).tolist(),
                        adjacent_unequal_target_n=different.sum(0).tolist(),
                        adjacent_equal_predictions_on_unequal_targets=adjacent_equal.tolist(),
                        error_margin_median_by_digit=[float(np.median(arrays['margin'][wrong[:, i], i]))
                            if wrong[:, i].any() else None for i in range(10)]))
    paired = read(HERE / 'results/depth_comparison.json')
    assert paired['passed'] and paired['state'] == 'complete' and len(paired['rows']) == 164
    for name, expected in paired['input_hashes'].items():
        assert sha(Path(name)) == expected
    for r in paired['rows']:
        key = r['task'], r['checkpoint'], r['T']
        two, three = records[(2,) + key], records[(3,) + key]
        for field in ['target', 'memory', 'positions']:
            assert np.array_equal(two[field], three[field])
        c2 = two['target'] == two['prediction']
        c3 = three['target'] == three['prediction']
        assert int((~c2.all(1) & c3.all(1)).sum()) == r['exact_rescued']
        assert int((c2.all(1) & ~c3.all(1)).sum()) == r['exact_regressed']
        assert ((~c2 & c3).sum(0)).tolist() == r['digit_rescued']
        assert ((c2 & ~c3).sum(0)).tolist() == r['digit_regressed']
    aggregates = []
    for task in TASKS:
        for cp in ['best', 'final']:
            for lo, hi in [(1, 192), (256, 2028), (2048, 131072)]:
                p = [r for r in paired['rows'] if r['task'] == task and
                     r['checkpoint'] == cp and lo <= r['T'] <= hi]
                summary = dict(task=task, checkpoint=cp, range=[lo, hi], horizons=len(p),
                    exact_improved_horizons=sum(r['exact_three'] > r['exact_two'] for r in p),
                    exact_regressed_horizons=sum(r['exact_three'] < r['exact_two'] for r in p),
                    rescued=sum(r['exact_rescued'] for r in p), regressed=sum(r['exact_regressed'] for r in p))
                for depth in [2, 3]:
                    rr = [r for r in rows if r['depth'] == depth and r['task'] == task and
                          r['checkpoint'] == cp and lo <= r['T'] <= hi]
                    summary[f'exact_mean_{depth}'] = sum(r['exact'] for r in rr) / (256 * len(rr))
                    summary[f'token_mean_{depth}'] = sum(r['token_correct'] for r in rr) / (2560 * len(rr))
                aggregates.append(summary)
    save(OUT / 'counts.json', rows)
    save(OUT / 'training.json', training)
    save(OUT / 'aggregates.json', aggregates)

    colors = {2: '#2474ad', 3: '#d76421'}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    for i, task in enumerate(TASKS):
        for depth in [2, 3]:
            for cp, style in [('best', '-'), ('final', '--')]:
                rr = [r for r in rows if r['task'] == task and r['depth'] == depth and r['checkpoint'] == cp]
                for j, field, denom in [(0, 'exact', 256), (1, 'token_correct', 2560)]:
                    axes[i, j].plot(GRID, [100 * r[field] / denom for r in rr], style,
                        color=colors[depth], linewidth=1.6, label=f'{depth} layers / {cp}')
        for j in range(2):
            ax = axes[i, j]
            ax.axvline(2028, color='gray', linestyle=':', label='Train max T=2028')
            ax.set(xscale='log', xlabel='T', ylabel='Accuracy (%)', ylim=(-2, 102),
                title=task + (' / exact string' if j == 0 else ' / digit'))
            ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(OUT / 'depth_accuracy.png', dpi=160); plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(17, 10), layout='constrained')
    for i, task in enumerate(TASKS):
        for j, (depth, cp) in enumerate([(2, 'best'), (3, 'best'), (2, 'final'), (3, 'final')]):
            rr = [r for r in rows if r['task'] == task and r['depth'] == depth and r['checkpoint'] == cp]
            ax = axes[i, j]
            im = ax.imshow(np.array([r['digit_errors'] for r in rr]) / 256 * 100,
                aspect='auto', cmap='magma', vmin=0, vmax=100)
            ticks = [k for k, t in enumerate(GRID) if t in [1, 3, 8, 14, 16, 48, 64, 128, 192, 256, 1024, 2048, 8192, 32768, 131072]]
            ax.set(xticks=range(10), xticklabels=range(1, 11), yticks=ticks,
                yticklabels=[str(GRID[k]) for k in ticks], xlabel='Digit (1-based)', ylabel='T (sampled grid)',
                title=f'{task}\n{depth} layers / {cp}')
            ax.axhline(GRID.index(2048)-.5, color='cyan', linestyle=':', linewidth=1)
    fig.colorbar(im, ax=axes, label='Digit error rate (%)', shrink=.6)
    fig.savefig(OUT / 'digit_errors.png', dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for ax, task in zip(axes, TASKS):
        for t in [t for t in training if t['task'] == task]:
            # Nonoverlapping 1000-step means of interval accuracies.
            intervals = t['intervals']
            ys = [np.mean([r['string_acc'] for r in intervals[k:k+10]]) * 100 for k in range(0, 500, 10)]
            ax.plot(range(1000, 50001, 1000), ys, color=colors[t['depth']], label=f"{t['depth']} layers")
        ax.set(title=task, xlabel='Training step', ylabel='String accuracy (1000-step mean, %)', ylim=(0, 100))
        ax.legend(); ax.grid(alpha=.2)
    fig.savefig(OUT / 'training.png', dpi=150); plt.close(fig)
    save(OUT / 'review.json', dict(passed=True, gpu_used=False, cells=328,
        campaigns=checks, all_archived_hashes_verified=True, current_main_runtime_unchanged=True,
        saved_predictions_recounted=True, paired_samples_and_counts_verified=True,
        script_sha256=sha(Path(__file__)),
        outputs={p.name: sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'review.json'},
        limitations='One seed; depth also increases parameter count. No new forward passes or hidden-state diagnosis. Aggregates weight the specified horizon grid equally, not the training distribution.'))
    print('Verified 328 cells, source/script/artifact/weight hashes, paired predictions and 50k completion.')
    for t in training:
        print(t['depth'], t['task'], 'best step', t['best']['step'], 'last5000', t['last5000_mean'])


if __name__ == '__main__':
    main()
