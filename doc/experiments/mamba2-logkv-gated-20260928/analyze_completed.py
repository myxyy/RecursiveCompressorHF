"""CPU audit of both hybrid tasks and paired LogKV / Mamba-2 controls."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

import torch

ARCHIVE = Path(__file__).resolve().parent
REPO = ARCHIVE.parents[2]
sys.path.insert(0, str(REPO))
from exp.copying import task as copying
from exp.selective_copying import task as selective
from exp.copying.evaluate import build_t_grid


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    baseline = REPO / 'doc/experiments/logkv-causal-conv-20260914/results'
    control_metrics = json.loads((baseline / 'metrics.json').read_text())
    all_reviews = {}
    for task, mamba_name in [(copying, 'mamba2-copying-20260926'),
                             (selective, 'mamba2-selective-copying-20260928')]:
        root = args.root / task.TASK_NAME
        dest = ARCHIVE / task.TASK_NAME
        dest.mkdir(parents=True, exist_ok=True)
        campaign = json.loads((root / 'campaign.json').read_text())
        assert campaign['state'] == 'complete' and campaign['task'] == task.TASK_NAME
        info = campaign['preflight']
        for name, expected in info['source'].items():
            assert digest(root / 'source' / name) == expected, name
        upstream = importlib.metadata.distribution('mamba-ssm')
        for name, expected in info['upstream_files'].items():
            assert digest(upstream.locate_file(name)) == expected, name
        run = root / 'exp' / task.TASK_NAME / 'hybrid'
        logs = [json.loads(s) for s in (run / 'train_log.jsonl').read_text().splitlines()]
        assert [r['step'] for r in logs] == list(range(100, 50001, 100))
        assert all(torch.isfinite(torch.tensor(r['loss'])) for r in logs)
        best = json.loads((run / 'best.json').read_text())
        assert best['step'] == max(logs, key=lambda r: (r['string_acc'], r['token_acc'], -r['ema_loss']))['step']
        model_config = json.loads((run / 'model/config.json').read_text())
        assert model_config['model_type'] == 'mamba2_logkv_gated'
        assert model_config['conv_kernel_size'] == 4 and model_config['d_conv'] == 4
        results, outputs, result_hashes = [], {}, {}
        gate_history = [r['mamba_gate'] for r in logs]
        assert all(len(g['values']) == 512 for g in gate_history)
        assert all(torch.isfinite(torch.tensor(g['values'])).all() for g in gate_history)
        assert all(torch.tensor(g['values']).abs().max() <= 1 for g in gate_history)
        serial_root = REPO / 'doc/experiments/mamba2-logkv-hybrid-20260928' / task.TASK_NAME
        serial_metrics = json.loads((serial_root / 'metrics.json').read_text())
        mamba_root = REPO / 'doc/experiments' / mamba_name / 'results'
        mamba_metrics = json.loads((mamba_root / 'metrics.json').read_text())
        for kind, directory in [('best', 'model_best'), ('final', 'model')]:
            path = root / 'results' / f'{kind}.json'
            raw = path.read_bytes()
            data = json.loads(raw)
            result_hashes[kind] = digest(path)
            outputs[kind] = data
            assert data['task'] == task.TASK_NAME and data['seed'] == 12345
            assert [c['T'] for c in data['cells']] == build_t_grid(17)
            for name, expected in data['checkpoint_sha256'].items():
                assert digest(run / directory / name) == expected
            from safetensors.torch import load_file
            raw_gate = load_file(run / directory / 'model.safetensors')['mamba_gate']
            master_gate = torch.tanh(raw_gate.float())
            logged_gate = next(r['mamba_gate'] for r in logs if r['step'] == (best['step'] if kind == 'best' else 50000))
            torch.testing.assert_close(master_gate, torch.tensor(logged_gate['values']), rtol=1e-6, atol=1e-7)
            eval_gate = torch.tanh(raw_gate.bfloat16().float())
            torch.testing.assert_close(eval_gate, torch.tensor(data['gate']['values']), rtol=1e-6, atol=1e-7)
            serial = json.loads(gzip.decompress((serial_root / f'{kind}.json.gz').read_bytes()))
            old = json.loads((baseline / task.TASK_NAME / f'digits_{kind}.json').read_text())
            mamba = json.loads(gzip.decompress((mamba_root / f'{kind}.json.gz').read_bytes()))
            generator = torch.Generator().manual_seed(12345)
            for cell, mc in zip(data['cells'], mamba['cells']):
                t = cell['T']
                assert cell['samples'] == 256 and t == mc['T']
                target = torch.tensor(cell['targets'])
                pred = torch.tensor(cell['predictions'])
                margins = torch.tensor(cell['margins'])
                assert target.shape == pred.shape == margins.shape == (256, 10)
                assert torch.isfinite(margins).all()
                batch = max(1, min(256, 2**19 // task.seq_len_for(t)))
                regen = []
                for start in range(0, 256, batch):
                    n = min(batch, 256-start)
                    ids, labels = task.make_batch(t, n, generator=generator, device='cpu')
                    regen.extend(labels[:, -10:].tolist())
                    if task is selective:
                        positions = ids[:, :t+9].nonzero()[:, 1].reshape(n, 10).tolist()
                        assert positions == old[str(t)][start//batch]['positions']
                assert regen == cell['targets'] == mc['targets']
                assert regen == [row for b in old[str(t)] for row in b['target']]
                matches = pred == target
                assert cell['token_correct'] == int(matches.sum())
                assert cell['string_correct'] == int(matches.all(-1).sum())
                assert cell['digit_errors'] == (~matches).sum(0).tolist()
                assert not ((margins > 0) & ~matches).any()
                assert not ((margins < 0) & matches).any()
                control = next(r for r in control_metrics if r['task'] == task.TASK_NAME
                               and r['checkpoint'] == kind and r['T'] == t)
                old_matches = target == torch.tensor([row for b in old[str(t)] for row in b['prediction']])
                assert int(old_matches.sum()) == control['token_correct']
                assert int(old_matches.all(-1).sum()) == control['string_correct']
                sc = next(c for c in serial['cells'] if c['T'] == t)
                assert sc['targets'] == cell['targets']
                sm = next(r for r in serial_metrics if r['checkpoint'] == kind and r['T'] == t)
                sc_matches = target == torch.tensor(sc['predictions'])
                assert int(sc_matches.sum()) == sm['token_correct']
                assert int(sc_matches.all(-1).sum()) == sm['string_correct']
                m = next(r for r in mamba_metrics if r['checkpoint'] == kind and r['T'] == t)
                mm = target == torch.tensor(mc['predictions'])
                assert int(mm.sum()) == m['token_correct']
                assert int(mm.all(-1).sum()) == m['string_correct']
                # Exact logical state size for the default BF16 evaluation.
                digits, length = 0, t + 20
                while length:
                    digits, length = digits + length % 4, length // 4
                expected_bytes = 1063936 + 2 * 3 * 512 * 2 * (digits + 1)
                assert cell['state_bytes_per_example'] == expected_bytes
                results.append(dict(checkpoint=kind, T=t, samples=256,
                    string_correct=cell['string_correct'], token_correct=cell['token_correct'],
                    string_acc=cell['string_correct']/256, token_acc=cell['token_correct']/2560,
                    digit_errors=cell['digit_errors'], state_bytes_per_example=expected_bytes,
                    logkv_string_acc=control['string_acc'], logkv_token_acc=control['token_acc'],
                    mamba_string_acc=m['string_acc'], mamba_token_acc=m['token_acc'],
                    serial_string_acc=sm['string_acc'], serial_token_acc=sm['token_acc']))
            (dest / f'{kind}.json.gz').write_bytes(gzip.compress(raw, mtime=0))
        assert all(a['targets'] == b['targets'] for a, b in zip(outputs['best']['cells'], outputs['final']['cells']))
        save(dest / 'metrics.json', results)
        for name in ['campaign.json', 'preflight.json']:
            shutil.copy2(root / name, dest / name)
        for name in ['train_log.jsonl', 'best.json', 'run_config.json']:
            shutil.copy2(run / name, dest / name)
        review = dict(passed=True, cells=82, best_step=best['step'], final_step=50000,
                      paired_targets_verified=True, placements_verified=task is selective,
                      sources_weights_predictions_verified=True,
                      source_manifest_sha256=digest(root / 'preflight.json'),
                      result_sha256=result_hashes, state_formula_verified=True,
                      logkv_metrics_sha256=digest(baseline / 'metrics.json'),
                      mamba_metrics_sha256=digest(mamba_root / 'metrics.json'),
                      serial_metrics_sha256=digest(serial_root / 'metrics.json'),
                      gate_logs_and_checkpoints_verified=True,
                      gate_summary={k: v['gate'] for k, v in outputs.items()},
                      final_training_metrics={k:v for k,v in logs[-1].items() if k != 'mamba_gate'}, gpu_work=False)
        save(dest / 'review.json', review)
        all_reviews[task.TASK_NAME] = review
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, metric in zip(axs, ['string_acc', 'token_acc']):
            for prefix, label, color in [('', 'Gated', '#CC79A7'), ('serial_', 'Serial', '#0072B2'), ('logkv_', 'LogKV', '#009E73'),
                                         ('mamba_', 'Mamba-2', '#D55E00')]:
                for kind, style in [('best', '-'), ('final', '--')]:
                    rows = [r for r in results if r['checkpoint'] == kind]
                    ax.plot([r['T'] for r in rows], [100*r[prefix+metric] for r in rows],
                            style, color=color, label=f'{label} {kind}')
            ax.axvline(2028, color='grey', linestyle=':')
            ax.set(xscale='log', xlabel='Horizon T', ylabel='Accuracy (%)', ylim=(-3, 104),
                   title='Exact match' if metric == 'string_acc' else 'Digit accuracy')
            ax.grid(alpha=.2)
        axs[1].legend(fontsize=8)
        fig.suptitle(f'Fixed-10 {task.TASK_NAME}: 256 paired examples, one seed; unequal parameter counts')
        fig.tight_layout()
        fig.savefig(dest / 'comparison.png', dpi=160)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot([r['step'] for r in logs], [g['abs_mean'] for g in gate_history], label='mean absolute gate')
        ax.plot([r['step'] for r in logs], [g['min'] for g in gate_history], label='minimum')
        ax.plot([r['step'] for r in logs], [g['max'] for g in gate_history], label='maximum')
        ax.set(xlabel='Training step', ylabel='tanh(gate)', title=task.TASK_NAME)
        ax.legend(); fig.tight_layout(); fig.savefig(dest / 'gate-history.png', dpi=160); plt.close(fig)
    save(ARCHIVE / 'review.json', dict(passed=True, cells=164, tasks=all_reviews,
                                     finished=datetime.now(timezone.utc).isoformat()))
    print('Verified both 50k runs, 164 cells, paired controls, weights, sources and logarithmic state.')


if __name__ == '__main__':
    main()
