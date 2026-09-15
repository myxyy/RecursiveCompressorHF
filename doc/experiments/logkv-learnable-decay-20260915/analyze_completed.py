"""Read-only CPU review of completed runs; write new analysis, preserve frozen results."""
import json
import math
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import HERE, ROOT, BASE_ROOT, MODES, run_dir, sha, save


def read(path):
    return json.loads(path.read_text())


def main():
    out = HERE / 'analysis'
    out.mkdir(exist_ok=True)
    results = HERE / 'results'
    baseline = HERE.parent / 'logkv-causal-conv-20260914/results'
    campaign = read(HERE / 'campaign.json')
    assert campaign == read(ROOT / 'campaign.json')
    assert campaign['state'] == 'complete-awaiting-review'
    assert not campaign['next_stage_queued']
    checked = {}

    def check(path, expected):
        actual = sha(path)
        assert actual == expected, str(path)
        checked[str(path)] = actual

    for name, expected in campaign['source_manifest']['files'].items():
        check(Path(campaign['source_root']) / name, expected)
    for group in ['scripts', 'prerequisite_hashes']:
        for name, expected in campaign[group].items():
            check(HERE / name, expected)
    check(ROOT / 'initial_model/model.safetensors', campaign['initial_weights_sha256'])
    check(BASE_ROOT / 'initial_model/model.safetensors', campaign['base_initial_sha256'])
    for folder in [results, baseline]:
        review = read(folder / 'review.json')
        assert review['passed'] and review['standard_cells'] == 164
        for name, expected in review['result_hashes'].items():
            check(folder / name, expected)
    metrics = read(results / 'metrics.json')
    old_metrics = read(baseline / 'metrics.json')
    lookup = {(r['task'], r['checkpoint'], r['T']): r for r in metrics}
    old_lookup = {(r['task'], r['checkpoint'], r['T']): r for r in old_metrics}
    paired = read(results / 'baseline_comparison.json')
    assert paired['passed']
    pair_lookup = {(r['task'], r['checkpoint'], r['T']): r for r in paired['rows']}
    coefficients = read(results / 'coefficients.json')
    summaries, histories, crossings = [], {}, []
    cells = 0
    for task in MODES:
        worker = read(results / task / 'worker.json')
        assert worker['state'] == 'complete' and worker['first_300_steps_match_benchmark']
        assert len(worker['commands']) == 3 and all(c['returncode'] == 0 for c in worker['commands'])
        history = [json.loads(line) for line in (results / task / 'beta_log.jsonl').read_text().splitlines()]
        assert [r['step'] for r in history] == [0] + list(range(100, 50001, 100))
        values = np.array([r['beta'] for r in history])
        assert values.shape == (501, 2, 8) and np.isfinite(values).all()
        histories[task] = values
        for layer in range(2):
            for head in range(8):
                negative = np.flatnonzero(values[:, layer, head] < 0)
                if len(negative):
                    crossings.append(dict(task=task, layer=layer+1, head=head+1,
                        first_recorded_negative_step=history[int(negative[0])]['step']))
        for cp in ['best', 'final']:
            weight = run_dir(task) / ('model_best' if cp == 'best' else 'model') / 'model.safetensors'
            check(weight, read(results / 'review.json')['weights'][f'{task}/{cp}'])
            state = load_file(weight, device='cpu')
            beta = torch.stack([state[f'layers.{i}.attention.level_decay'] for i in range(2)])
            row = next(r for r in coefficients['rows'] if (r['task'], r['checkpoint']) == (task, cp))
            assert beta.tolist() == row['beta'] == history[row['step']//100]['beta']
            assert beta.bfloat16().float().tolist() == row['beta_bf16_evaluation']
            old = read(baseline / task / f'digits_{cp}.json')
            new = read(results / task / f'digits_{cp}.json')
            assert old.keys() == new.keys() and len(new) == 41
            for horizon, batches in new.items():
                def flat(records, field):
                    return np.array([x for b in records for x in b[field]])
                for field in ['target', 'memory', 'positions']:
                    assert np.array_equal(flat(batches, field), flat(old[horizon], field))
                target = flat(batches, 'target')
                correct = flat(batches, 'prediction') == target
                previous = flat(old[horizon], 'prediction') == target
                key = task, cp, int(horizon)
                for correctness, record in [(correct, lookup[key]), (previous, old_lookup[key])]:
                    assert correctness.shape == (256, 10)
                    assert int(correctness.sum()) == record['token_correct']
                    assert int(correctness.all(axis=1).sum()) == record['string_correct']
                    assert (~correctness).sum(axis=0).tolist() == record['digit_errors']
                    assert correctness.mean() == record['token_acc']
                    assert correctness.all(axis=1).mean() == record['string_acc']
                    cells += 1
                pr = pair_lookup[key]
                assert int((~previous.all(1) & correct.all(1)).sum()) == pr['exact_rescued']
                assert int((previous.all(1) & ~correct.all(1)).sum()) == pr['exact_regressed']
            rows = [r for r in metrics if (r['task'], r['checkpoint']) == (task, cp)]
            summaries.append(dict(task=task, checkpoint=cp,
                perfect_horizons=sum(r['string_correct'] == 256 for r in rows),
                digit_errors=np.sum([r['digit_errors'] for r in rows], axis=0).tolist(),
                nonperfect=[dict(T=r['T'], exact=r['string_correct']) for r in rows if r['string_correct'] < 256]))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
    for ti, task in enumerate(MODES):
        for layer in range(2):
            ax = axes[ti, layer]
            for head in range(8):
                ax.plot(np.arange(501)*100, histories[task][:, layer, head], label=f'H{head+1}')
            ax.axhline(math.log(4), color='gray', ls='--', lw=1)
            ax.axhline(0, color='black', lw=1)
            ax.set(title=f'{task}, layer {layer+1}', xlabel='Training step', ylabel='Beta (negative = amplification)')
            ax.grid(alpha=.2)
    axes[0, 1].legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'beta_history.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for ti, task in enumerate(MODES):
        for ci, field in enumerate(['string_acc', 'token_acc']):
            ax = axes[ti, ci]
            for table, label, color in [(old_metrics, 'fixed', 'gray'), (metrics, 'learned', 'tab:blue')]:
                for cp, style in [('best', '-'), ('final', '--')]:
                    rows = sorted([r for r in table if (r['task'], r['checkpoint']) == (task, cp)], key=lambda r:r['T'])
                    ax.plot([r['T'] for r in rows], [r[field]*100 for r in rows], style, color=color, label=f'{label} {cp}')
            ax.set(xscale='log', ylim=(-2,102), title=f'{task}: {field}', xlabel='T', ylabel='Accuracy (%)')
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'accuracy.png', dpi=160)
    plt.close(fig)
    save(out / 'review.json', dict(passed=True, recounted_cells=cells, hashes=checked,
        summaries=summaries, first_negative_snapshots=crossings,
        campaign_finished=campaign['finished'], gpu_work_started=False,
        limitations='One seed, 41 sampled horizons through T131072. Existing autocast precision confound; no causal intervention on beta and no 16M evaluation.'))
    print(json.dumps(dict(passed=True, recounted_cells=cells, summaries=summaries, crossings=crossings), indent=2))


if __name__ == '__main__':
    main()
