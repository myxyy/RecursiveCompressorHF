"""CPU completion review and plots; never modifies hash-indexed run artifacts."""
import json
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import HERE, ROOT, BASE_ROOT, SOURCE, INITIAL, BASE_INITIAL, MODES, sha, save
from compare_baseline import compare_records


def read(p):
    return json.loads(p.read_text())


def main():
    out = HERE / 'analysis'; out.mkdir(exist_ok=True)
    directories = {'baseline': HERE.parent / 'logkv-no-position-main-20260913', 'conv': HERE}
    roots = {'baseline': BASE_ROOT, 'conv': ROOT}
    campaign = read(HERE / 'campaign.json')
    assert campaign == read(ROOT / 'campaign.json')
    assert campaign['state'] == 'complete-awaiting-review' and not campaign['next_stage_queued']
    assert campaign['whole_gpu_batch_hours'] < 7.5
    for name, digest in campaign['scripts'].items():
        assert sha(HERE / name) == digest
    for name, digest in campaign['prerequisite_hashes'].items():
        assert sha(HERE / name) == digest
    for name, digest in campaign['source_manifest']['files'].items():
        assert sha(SOURCE / name) == digest
        assert sha(HERE.parents[2] / name) == digest
    assert sha(INITIAL) == campaign['initial_weights_sha256']
    assert sha(BASE_INITIAL) == campaign['base_initial_sha256']
    main_commit = subprocess.check_output(['git', 'rev-parse', 'main'], text=True).strip()
    assert main_commit.startswith('d707675')
    grid = sorted(set(range(1, 15)) | {2**k for k in range(4, 18)} |
                  {3*2**(k-1) for k in range(4, 17)})
    data, rows, training, audit_hashes = {}, [], [], {}
    for variant, directory in directories.items():
        result = directory / 'results'; review = read(result / 'review.json')
        assert review['passed'] and review['standard_cells'] == 164
        audit_hashes[variant] = sha(result / 'review.json')
        for name, digest in review['result_hashes'].items():
            assert sha(result / name) == digest
        metrics = read(result / 'metrics.json'); assert len(metrics) == 164
        for task in MODES:
            local = result / task
            cfg = read(local / 'run_config.json')
            assert cfg['num_layers'] == 2 and not cfg['phase_emb']
            assert cfg.get('conv_kernel_size', 0) == (4 if variant == 'conv' else 0)
            assert cfg['self_slot'] and cfg['gated_attention']
            worker = read(local / 'worker.json')
            assert worker['state'] == 'complete' and len(worker['commands']) == 3
            assert all(c['returncode'] == 0 for c in worker['commands'])
            intervals = [json.loads(s) for s in (local / 'train_log.jsonl').read_text().splitlines()]
            intervals = [r for r in intervals if 'loss' in r]
            assert [r['step'] for r in intervals] == list(range(100, 50001, 100))
            assert all(np.isfinite(r[k]) for r in intervals for k in ['loss', 'ema_loss', 'token_acc', 'string_acc'])
            best = read(local / 'best.json')
            assert best['step'] == max(intervals, key=lambda r: (
                r['string_acc'], r['token_acc'], -r['ema_loss']))['step']
            perfect = [r['step'] for r in intervals if r['string_acc'] == 1]
            imperfect = [r['step'] for r in intervals if r['string_acc'] < 1]
            training.append(dict(variant=variant, task=task, best=best,
                first_perfect_interval=min(perfect) if perfect else None,
                last_imperfect_interval=max(imperfect) if imperfect else None,
                final_interval=intervals[-1], intervals=intervals))
            for cp, folder in [('best', 'model_best'), ('final', 'model')]:
                weights = roots[variant] / 'exp' / task / cfg['run_name'] / folder / 'model.safetensors'
                assert sha(weights) == review['weights'][f'{task}/{cp}'] == worker['weights'][cp]
                saved = read(local / f'digits_{cp}.json'); data[variant, task, cp] = saved
                assert sorted(map(int, saved)) == grid
                for t in grid:
                    arrays = {k: np.array([v for b in saved[str(t)] for v in b[k]])
                              for k in ['target', 'prediction', 'memory', 'positions', 'margin']}
                    assert all(a.shape == (256, 10) for a in arrays.values())
                    assert np.array_equal(arrays['target'], arrays['memory'])
                    correct = arrays['target'] == arrays['prediction']
                    assert np.isfinite(arrays['margin']).all()
                    exact, tok = int(correct.all(1).sum()), int(correct.sum())
                    errors = (~correct).sum(0).tolist()
                    metric = next(r for r in metrics if r['mode'] == task and r['checkpoint'] == cp and r['T'] == t)
                    assert metric['string_correct'] == exact and metric['token_correct'] == tok
                    assert metric['digit_errors'] == errors and metric['string_acc'] == exact/256
                    assert metric['token_acc'] == tok/2560
                    rows.append(dict(variant=variant, task=task, checkpoint=cp, T=t,
                        exact=exact, token_correct=tok, n=256, digit_errors=errors,
                        margin_min=float(arrays['margin'].min()), margin_median=float(np.median(arrays['margin']))))
    paired = read(HERE / 'results/baseline_comparison.json')
    assert paired['passed'] and paired['state'] == 'complete'
    for name, digest in paired['input_hashes'].items():
        assert sha(Path(name)) == digest
    for task in MODES:
        for cp in ['best', 'final']:
            recalculated = compare_records(data['baseline', task, cp], data['conv', task, cp])
            expected = [{k: v for k, v in r.items() if k not in ['task', 'checkpoint']}
                        for r in paired['rows'] if r['task'] == task and r['checkpoint'] == cp]
            assert recalculated == expected
    conv_review = read(HERE / 'results/review.json')
    same_copy_weights = conv_review['weights']['copying/best'] == conv_review['weights']['copying/final']
    assert same_copy_weights and data['conv', 'copying', 'best'] == data['conv', 'copying', 'final']
    copying = [r for r in rows if r['variant'] == 'conv' and r['task'] == 'copying']
    assert len(copying) == 82 and all(r['exact'] == 256 and not any(r['digit_errors']) for r in copying)
    save(out / 'counts.json', rows); save(out / 'training.json', training)
    colors = {'baseline': '#2474ad', 'conv': '#d76421'}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for i, task in enumerate(MODES):
        for variant in directories:
            for cp, style in [('best', '-'), ('final', '--')]:
                rr = [r for r in rows if r['variant'] == variant and r['task'] == task and r['checkpoint'] == cp]
                for j, field, divisor in [(0, 'exact', 256), (1, 'token_correct', 2560)]:
                    axes[i, j].plot(grid, [100*r[field]/divisor for r in rr], style,
                        color=colors[variant], label=f'{variant} / {cp}', linewidth=1.5)
        for j in range(2):
            ax = axes[i, j]; ax.axvline(2028, color='gray', linestyle=':', label='Train max T=2028')
            ax.set(xscale='log', xlabel='T', ylabel='Accuracy (%)', ylim=(-2, 102),
                   title=task + (' / exact string' if j == 0 else ' / digit'))
            ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.savefig(out / 'comparison.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for ax, task in zip(axes, MODES):
        for rec in [r for r in training if r['task'] == task]:
            intervals = rec['intervals']
            y = [np.mean([r['string_acc'] for r in intervals[k:k+10]])*100 for k in range(0, 500, 10)]
            ax.plot(range(1000, 50001, 1000), y, color=colors[rec['variant']], label=rec['variant'])
        ax.set(title=task, xlabel='Training step', ylabel='String accuracy (1000-step mean, %)', ylim=(-2, 102))
        ax.legend(); ax.grid(alpha=.2)
    fig.savefig(out / 'training.png', dpi=150); plt.close(fig)
    save(out / 'review.json', dict(passed=True, cpu_only=True, cells_recounted=328,
        frozen_source_script_prerequisite_hashes_verified=True, result_review_hashes=audit_hashes,
        checkpoint_hashes_verified=True, paired_samples_verified=True, main_unchanged=main_commit,
        copying_best_final_same_weights=same_copy_weights, copying_samples_per_checkpoint=41*256,
        copying_min_margin=min(r['margin_min'] for r in copying),
        script_sha256=sha(Path(__file__)),
        files={p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name != 'review.json'},
        limitation='One seed; best/final Copying are the same checkpoint and same evaluation data. Only the sampled 41 horizons through T131072 were evaluated; no 16M extension or new inference in this review.'))
    print('All completion checks passed. Copying perfect on 41 horizons; best/final share weights/data.')
    print('Minimum Copying margin:', min(r['margin_min'] for r in copying))


if __name__ == '__main__':
    main()
