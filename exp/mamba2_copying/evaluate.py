"""Evaluate both saved checkpoints and retain predictions for independent audit."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from exp.copying.evaluate import build_t_grid
from exp.copying import task
from models.mamba2.modeling import Mamba2LM


def evaluate(model_dir, samples=256, max_exp=17):
    device = torch.device('cuda:0')
    model = Mamba2LM.from_pretrained(model_dir).to(device=device, dtype=torch.bfloat16).eval()
    generator = torch.Generator().manual_seed(12345)
    cells = []
    with torch.inference_mode():
        for horizon in build_t_grid(max_exp):
            start = time.monotonic()
            batch_size = max(1, min(samples, 2**19 // task.seq_len_for(horizon)))
            expected, predicted, margins = [], [], []
            state_bytes = None
            for offset in range(0, samples, batch_size):
                b = min(batch_size, samples-offset)
                ids, labels = task.make_batch(horizon, b, generator=generator, device='cpu')
                hidden, tail = None, None
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    for i in range(0, ids.shape[1], 8192):
                        logits, hidden = model.step(ids[:, i:i+8192].to(device), hidden)
                        tail = logits[:, -10:] if tail is None else torch.cat([tail, logits[:, -10:]], 1)[:, -10:]
                logits = tail.float().cpu()
                if not torch.isfinite(logits).all():
                    raise RuntimeError(f'Nonfinite logits at T={horizon}')
                target = labels[:, -10:]
                correct = logits.gather(-1, target[...,None]).squeeze(-1)
                wrong = logits.clone().scatter_(-1, target[...,None], float('-inf')).amax(-1)
                expected.extend(target.tolist())
                predicted.extend(logits.argmax(-1).tolist())
                margins.extend((correct-wrong).tolist())
                state_bytes = sum(t.numel()*t.element_size() for layer in hidden for t in layer)//b
                # Ensure the cache owns only the fixed-size tensors, not input-chunk storage.
                assert all(t.untyped_storage().nbytes() == t.numel()*t.element_size()
                           for layer in hidden for t in layer)
            target, pred = torch.tensor(expected), torch.tensor(predicted)
            matches = target == pred
            cell = dict(T=horizon, samples=samples, token_correct=int(matches.sum()),
                        string_correct=int(matches.all(-1).sum()),
                        digit_errors=(~matches).sum(0).tolist(), targets=expected,
                        predictions=predicted, margins=margins, state_bytes_per_example=state_bytes,
                        seconds=time.monotonic()-start)
            cells.append(cell)
            print(f'T={horizon} exact={cell["string_correct"]}/{samples} '
                  f'tokens={cell["token_correct"]}/{samples*10} seconds={cell["seconds"]:.2f}', flush=True)
    weights = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in model_dir.glob('*.safetensors')}
    return dict(model_dir=str(model_dir), precision='bf16 weights and autocast',
                seed=12345, checkpoint_sha256=weights, cells=cells)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=256)
    p.add_argument('--max-exp', type=int, default=17)
    args = p.parse_args()
    torch.set_float32_matmul_precision('high')
    result = evaluate(args.model_dir,args.samples,args.max_exp)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
