"""Frozen Selective evaluator with observations from the actual generated samples."""
import argparse
import importlib.util
import json
import shutil
import sys

import torch

from common import (MODES, ROOT, SOURCE, TASK_NAME, bind_selective_task, name,
                    run_dir, save, task_identity)

sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE / 'exp/copying'))
task = bind_selective_task()
spec = importlib.util.spec_from_file_location('standard_evaluate', SOURCE / 'exp/copying/evaluate.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
assert ev.TASK_NAME == TASK_NAME
assert ev.make_batch is task.make_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--checkpoint', choices=['best', 'final'], required=True)
    args = parser.parse_args()
    out = ROOT / args.mode
    records = {}
    current = {}
    pending = None
    original_batch = ev.make_batch
    original_score = ev.score_logits
    original_horizon = ev.eval_horizon
    original_argv = sys.argv

    def batch(T, *batch_args, **batch_kwargs):
        nonlocal pending
        assert pending is None
        inputs, labels = original_batch(T, *batch_args, **batch_kwargs)
        # Observe this exact batch; no extra generator calls or changed batch sizing.
        prefix = inputs[:, :T + 9]
        data_mask = (prefix >= 1) & (prefix <= 8)
        assert (data_mask.sum(-1) == 10).all()
        positions = data_mask.nonzero(as_tuple=False)[:, 1].reshape(inputs.shape[0], 10)
        memory = prefix.gather(1, positions)
        assert torch.equal(memory, labels[:, -10:])
        assert (inputs[:, T + 9:] == 9).all()
        pending = dict(positions=positions.cpu().tolist(), memory=memory.cpu().tolist())
        return inputs, labels

    def score(logits, labels):
        nonlocal pending
        assert pending is not None
        result = original_score(logits, labels)
        answer_logits = logits[:, -10:].float()
        target = labels[:, -10:]
        correct = answer_logits.gather(-1, target[..., None]).squeeze(-1)
        rivals = answer_logits.clone()
        rivals.scatter_(-1, target[..., None], float('-inf'))
        records[str(current['T'])].append(dict(
            target=target.cpu().tolist(), prediction=answer_logits.argmax(-1).cpu().tolist(),
            margin=(correct - rivals.max(-1).values).cpu().tolist(), **pending))
        pending = None
        return result

    def horizon(model, T, *horizon_args, **horizon_kwargs):
        assert pending is None
        assert str(T) not in records
        current['T'] = T
        records[str(T)] = []
        result = original_horizon(model, T, *horizon_args, **horizon_kwargs)
        assert pending is None
        return result

    folder = 'model_best' if args.checkpoint == 'best' else 'model'
    config = json.loads((run_dir(args.mode) / folder / 'config.json').read_text())
    assert config['aligned_rope'] == (args.mode == 'aligned')
    assert config['retrieval_rope'] == (args.mode == 'local-control')
    assert config['aligned_rope_scale'] == config['retrieval_rope_scale'] == 1.0
    assert config['self_slot'] and not config['phase_emb']
    try:
        ev.make_batch = batch
        ev.score_logits = score
        ev.eval_horizon = horizon
        sys.argv = [str(SOURCE / 'exp/selective-copying/evaluate.py'),
                    '--run-name', name(args.mode), '--samples', '256', '--max-t-exp', '13',
                    '--seed', '12345', '--precision', 'bf16', '--checkpoint', args.checkpoint,
                    '--device', '0']
        ev.main()
        assert pending is None
        save(out / f'digits_{args.checkpoint}.json', records)
        payload = json.loads((run_dir(args.mode) / 'results.json').read_text())
        payload.update(task_identity())
        save(run_dir(args.mode) / 'results.json', payload)
        save(out / f'results_{args.checkpoint}.json', payload)
        shutil.copy2(run_dir(args.mode) / 'plot.png', out / f'plot_{args.checkpoint}.png')
        save(out / f'evaluation_audit_{args.checkpoint}.json', dict(
            passed=True, mode=args.mode, checkpoint=args.checkpoint, **task_identity(),
            generated_samples_observed=True, extra_random_draws=False,
            seed=12345, samples_per_horizon=256, horizons=list(map(int, records)),
            token_budget=ev.TOKEN_BUDGET, chunk_len=ev.CHUNK_LEN))
    finally:
        ev.make_batch = original_batch
        ev.score_logits = original_score
        ev.eval_horizon = original_horizon
        sys.argv = original_argv


if __name__ == '__main__':
    main()
