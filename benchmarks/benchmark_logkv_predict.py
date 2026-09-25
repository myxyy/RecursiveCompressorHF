"""Compare one-token step/predict on the same prefix and supplied tokens.

Example: .venv/bin/python -m benchmarks.benchmark_logkv_predict --target lm --mode autocast
Prefill, correctness checks and warmup are excluded from synchronized timings.
No training or text sampling is performed. GPU use is confined to --device.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import torch

from models.logkv.configuration import LogKVConfig
from models.logkv.attention import LogKV
from models.logkv.modeling import LogKVLM


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', choices=['attention', 'lm'], default='attention')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--mode', choices=['float32', 'bfloat16', 'autocast'], default='bfloat16')
    p.add_argument('--checkpoint', type=Path, help='Optional trained LM directory (local only)')
    p.add_argument('--prefill', type=int, default=2048)
    p.add_argument('--tokens', type=int, default=256)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--dim', type=int, default=1024)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--layers', type=int, default=16)
    p.add_argument('--ff', type=int, default=3072)
    p.add_argument('--vocab-size', type=int, default=32000)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if min(args.prefill, args.tokens, args.repeats, args.batch_size) < 1:
        p.error('prefill, tokens, repeats and batch-size must be positive')
    if args.checkpoint is not None and args.target != 'lm':
        p.error('--checkpoint requires --target lm')
    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.manual_seed(20260916)
    dtype = torch.bfloat16 if args.mode == 'bfloat16' else torch.float32
    if args.target == 'lm':
        if args.checkpoint is not None:
            model = LogKVLM.from_pretrained(args.checkpoint, local_files_only=True)
        else:
            model = LogKVLM(LogKVConfig(d_model=args.dim, num_heads=args.heads,
                num_layers=args.layers, d_ff=args.ff, vocab_size=args.vocab_size,
                conv_kernel_size=4, self_slot=True, gated_attention=True))
        config = model.config.to_dict()
        x = torch.randint(model.config.vocab_size, (args.batch_size, args.prefill + args.tokens), device=device)
    else:
        model = LogKV(args.dim, 4, num_heads=args.heads, self_slot=True, gated_attention=True)
        config = dict(dim=args.dim, num_heads=args.heads, chunk_size=4,
                      self_slot=True, gated_attention=True)
        x = torch.randn(args.batch_size, args.prefill + args.tokens, args.dim, device=device, dtype=dtype)
    model = model.to(device=device, dtype=dtype).eval()
    sequence = x[:, args.prefill:].unbind(1)

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def run(kind, hidden, count=None):
        for token in sequence[:count]:
            if kind == 'step':
                out, hidden = model.step(token.unsqueeze(1), hidden)
            else:
                out, hidden = model.predict(token, hidden)
        return out, hidden

    precision = (torch.autocast(device.type, dtype=torch.bfloat16)
                 if args.mode == 'autocast' else contextlib.nullcontext())
    with torch.inference_mode(), precision:
        prefix = None
        # Limit transient logits allocation even for long prompts.
        for start in range(0, args.prefill, 256):
            _, prefix = model.step(x[:, start:min(start+256, args.prefill)], prefix)
        fast = old = prefix
        max_error = squared_error = squared_ref = numel = top1_equal = top1_total = 0
        for token in sequence:
            ref, old = model.step(token.unsqueeze(1), old)
            out, fast = model.predict(token, fast)
            ref = ref[:, 0].float()
            error = out.float() - ref
            if not torch.isfinite(out).all():
                raise RuntimeError('Nonfinite predict output')
            max_error = max(max_error, error.abs().max().item())
            squared_error += error.square().sum().item()
            squared_ref += ref.square().sum().item()
            numel += error.numel()
            if args.target == 'lm':
                top1_equal += (out.argmax(-1) == ref.argmax(-1)).sum().item()
                top1_total += args.batch_size
        del old, fast, ref, out, error
        run('step', prefix, min(args.tokens, 32))
        run('predict', prefix, min(args.tokens, 32))
        timings = {'step': [], 'predict': []}
        for repeat in range(args.repeats):
            for kind in (('step', 'predict') if repeat % 2 == 0 else ('predict', 'step')):
                sync()
                start = time.perf_counter()
                run(kind, prefix)
                sync()
                timings[kind].append(1000 * (time.perf_counter() - start) / args.tokens)
    medians = {key: statistics.median(values) for key, values in timings.items()}
    checkpoint_hashes = None
    if args.checkpoint is not None:
        checkpoint_hashes = {}
        for path in sorted(args.checkpoint.glob('*.safetensors')):
            with path.open('rb') as f:
                checkpoint_hashes[path.name] = hashlib.file_digest(f, 'sha256').hexdigest()
    result = dict(
        args={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        config=config, torch=torch.__version__, python=platform.python_version(),
        checkpoint_sha256=checkpoint_hashes,
        device=torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
        source_sha256={name: hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
                       for name in ['models/logkv/attention.py', 'models/logkv/modeling.py',
                                    'benchmarks/benchmark_logkv_predict.py']},
        milliseconds_per_token=timings, median_ms_per_token=medians,
        speedup=medians['step']/medians['predict'],
        correctness=dict(max_abs_error=max_error, rmse=(squared_error/numel)**0.5,
                         relative_rmse=(squared_error/max(squared_ref, 1e-30))**0.5,
                         top1_equal=top1_equal, top1_total=top1_total),
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ['median_ms_per_token', 'speedup', 'correctness']}, indent=2))


if __name__ == '__main__':
    main()
