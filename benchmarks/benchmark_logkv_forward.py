"""Compare old and stateless training forwards using identical synthetic data.

Single block: python -m benchmarks.benchmark_logkv_forward --target block
Pipeline: torchrun --standalone --nproc_per_node=2 --module
          benchmarks.benchmark_logkv_forward --target pipeline

Times include forward/backward and gradient clearing, but exclude optimizer
updates, loading, sampling and checkpoint I/O. No weights are updated.
"""
import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import torch
import torch.distributed as dist

from models.logkv.attention import CausalConvBlock, LogKV, LogKVBlock
from models.logkv.configuration import LogKVConfig
from models.logkv.pipeline import LogKVLMPipelineStage
from training.train_logkv_pipeline import PipelineEngine


@contextmanager
def execution_path(kind):
    # Recreate the previous pipeline forward exactly, without changing weights,
    # attention checkpointing, slot selection or arithmetic inside each level.
    with ExitStack() as stack:
        if kind == 'step':
            stack.enter_context(patch.object(LogKV, 'forward', lambda self, x: self.step(x)[0]))
            stack.enter_context(patch.object(CausalConvBlock, 'forward', lambda self, x: self.step(x)[0]))
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', choices=['block', 'pipeline'], default='block')
    parser.add_argument('--dim', type=int, default=512)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--ff', type=int, default=1536)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--vocab', type=int, default=1024)
    parser.add_argument('--length', type=int, default=2047)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--microbatches', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--no-level-bias', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.dim, args.heads, args.ff, args.layers, args.vocab, args.length,
           args.batch_size, args.microbatches, args.warmup, args.repeats, args.iterations) < 1:
        parser.error('counts must be positive')
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(20261029 + rank)
    if args.target == 'pipeline':
        if args.batch_size % args.microbatches or args.microbatches < world:
            parser.error('batch must divide by microbatches; microbatches must be >= world size')
        dist.init_process_group('nccl')
    elif world != 1:
        parser.error('block target uses one process')
    try:
        options = dict(num_heads=args.heads, self_slot=True, gated_attention=True,
                       conv_kernel_size=4, no_level_bias=args.no_level_bias)
        if args.target == 'block':
            model = LogKVBlock(args.dim, 4, args.ff, **options).to(device).train()
            x = torch.randn(args.batch_size, args.length, args.dim, device=device,
                            requires_grad=True)

            def run():
                model.zero_grad(set_to_none=True)
                x.grad = None
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    output = model(x)
                    loss = output.float().square().mean()
                loss.backward()
                return loss.detach()
        else:
            config = LogKVConfig(vocab_size=args.vocab, d_model=args.dim, d_ff=args.ff,
                                 num_layers=args.layers, chunk_size=4, **options)
            info = LogKVLMPipelineStage.split_config(args.layers, world)[rank]
            model = LogKVLMPipelineStage(config, **info).to(device).train()
            generator = torch.Generator().manual_seed(20261029)
            ids = torch.randint(args.vocab, (args.batch_size, args.length), generator=generator)
            labels = torch.randint(args.vocab, ids.shape, generator=generator)
            labels[:, :args.length // 3] = -100
            engine = PipelineEngine(model, device, args.microbatches, 'bf16')

            def run():
                model.zero_grad(set_to_none=True)
                return engine.backward([(ids, labels)])

        # Establish numerical parity before warmup or timing. Each mode starts
        # with exactly the same parameters and deterministic supplied data.
        checks = {}
        reference = None
        for kind in ['step', 'forward']:
            with execution_path(kind):
                loss = float(run())
            grads = {name: p.grad.detach().float().cpu().clone()
                     for name, p in model.named_parameters() if p.grad is not None}
            if not all(torch.isfinite(g).all() for g in grads.values()):
                raise RuntimeError('Nonfinite gradient')
            checks[kind + '_loss'] = loss
            if reference is None:
                reference = grads
            else:
                assert reference.keys() == grads.keys()
                checks['gradient_max_abs_difference'] = max(
                    (grads[name] - reference[name]).abs().max().item() for name in grads)
                checks['gradient_relative_l2_difference'] = (
                    sum((grads[n] - reference[n]).double().square().sum().item() for n in grads)
                    / max(1e-30, sum(g.double().square().sum().item() for g in reference.values()))) ** .5
                for name in grads:
                    torch.testing.assert_close(grads[name], reference[name], atol=2e-3, rtol=.05)
        del grads, reference

        for kind in ['step', 'forward']:
            with execution_path(kind):
                for _ in range(args.warmup):
                    run()
        times = dict(step=[], forward=[])
        peaks = dict(step=[], forward=[])
        for repeat in range(args.repeats):
            for kind in (['step', 'forward'] if repeat % 2 == 0 else ['forward', 'step']):
                model.zero_grad(set_to_none=True)
                if args.target == 'block':
                    x.grad = None
                if args.target == 'pipeline':
                    dist.barrier()
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                with execution_path(kind):
                    started = time.perf_counter()
                    for _ in range(args.iterations):
                        run()
                    torch.cuda.synchronize(device)
                    elapsed = (time.perf_counter() - started) / args.iterations
                peak = torch.cuda.max_memory_allocated(device)
                if args.target == 'pipeline':
                    value = torch.tensor(elapsed, device=device, dtype=torch.float64)
                    dist.all_reduce(value, op=dist.ReduceOp.MAX)
                    elapsed = value.item()
                times[kind].append(elapsed * 1000)
                peaks[kind].append(peak / 2**20)
        local = dict(rank=rank, checks=checks, milliseconds=times, peak_allocated_mib=peaks)
        ranks = [local]
        if args.target == 'pipeline':
            ranks = [None] * world
            dist.all_gather_object(ranks, local)
        if rank == 0:
            result = dict(arguments={**vars(args), 'output': str(args.output)},
                          torch_version=torch.__version__, gpu=torch.cuda.get_device_name(device),
                          world_size=world, precision='fp32 master + bf16 autocast',
                          scope='forward/backward/zero_grad; no optimizer update or I/O', ranks=ranks,
                          median_ms={k: statistics.median(v) for k, v in times.items()})
            result['speedup'] = result['median_ms']['step'] / result['median_ms']['forward']
            result['source_sha256'] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [
                'models/logkv/attention.py', 'models/logkv/modeling.py',
                'models/logkv/pipeline.py', 'training/train_logkv_pipeline.py', __file__]}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(dict(median_ms=result['median_ms'], speedup=result['speedup'],
                                  peak_mib={k: max(v) for k, v in peaks.items()})), flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
