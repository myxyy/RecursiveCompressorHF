"""1F1B pipeline training for LogKVLM, one stage per GPU (no DDP replicas).

torchrun --standalone --nproc_per_node=6 --module training.train_logkv_pipeline \
    --run-name pipeline --batch-size 12 --n-microbatches 12

Checkpoints contain model/ (HF weights + tokenizer for inference/predict_stream.py) and
per-stage optimizer/RNG state. --resume latest resumes the exact data cursor;
--start-checkpoint loads weights only, including ordinary LogKV DDP models.
"""
import contextlib
from datetime import timedelta
import fcntl
import itertools
import json
import os
from pathlib import Path
import random
import shutil
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import Schedule1F1B
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from models.logkv.configuration import LogKVConfig
from data_pipeline.dataset import prepare_all_datasets
from models.logkv.pipeline import LogKVLMPipelineStage, export_model
from training.train_logkv import (build_parser, get_data_dir, split_params_for_muon,
                         SAMPLE_PROMPTS, _CMD_MAP, CMD_NONE, CMD_PAUSE,
                         CMD_RESUME, CMD_SAVE_AND_EXIT)


def parse_args(argv=None):
    p = build_parser()
    p.description = 'LogKVLM pipeline parallel training (one stage per GPU)'
    p.set_defaults(batch_size=12)
    # In a pipeline the batch is shared, not multiplied by the GPU count.
    for action in p._actions:
        if action.dest == 'batch_size':
            action.help = 'batch per pipeline schedule; divisible by n-microbatches (not per GPU)'
        elif action.dest == 'resume':
            action.help = "'latest', a checkpoint name in this run, or a checkpoint path"
    p.add_argument('--n-microbatches', type=int, default=12)
    p.add_argument('--stage-layer-split', type=lambda s: [int(x) for x in s.split(',')],
                   help='comma-separated layer counts, e.g. 2,2,3,3,3,3 (default: even split)')
    p.add_argument('--start-checkpoint', help='HF LogKV model directory or checkpoint containing model/; weights only')
    p.add_argument('--precision', choices=['bf16', 'fp32'], default='bf16')
    p.add_argument('--num-workers', type=int, default=2, help='DataLoader workers per rank')
    p.add_argument('--cache-build-workers', type=int, default=32)
    p.add_argument('--control-file', default='control.cmd')
    p.add_argument('--sample-max-new-tokens', type=int, default=80)
    args = p.parse_args(argv)
    positive = ['batch_size', 'grad_accum', 'n_microbatches', 'num_epochs',
                'checkpoint_interval', 'max_checkpoints', 'log_interval',
                'cache_build_workers', 'sample_max_new_tokens']
    nonnegative = ['warmup', 'max_steps', 'sample_interval', 'num_workers']
    if any(getattr(args, key) < 1 for key in positive) or any(getattr(args, key) < 0 for key in nonnegative):
        p.error('invalid nonpositive count or negative interval')
    if args.context_length < 2 or args.lr <= 0 or args.grad_clip <= 0:
        p.error('context-length must be >=2, lr and grad-clip must be positive')
    if args.batch_size % args.n_microbatches:
        p.error('batch-size must be divisible by n-microbatches')
    if args.n_microbatches < int(os.environ.get('WORLD_SIZE', 1)):
        p.error('1F1B requires n-microbatches >= number of stages')
    if args.resume and args.start_checkpoint:
        p.error('--resume and --start-checkpoint are mutually exclusive')
    if Path(args.run_name).name != args.run_name or args.run_name in ('.', '..'):
        p.error('run-name must be a single directory name')
    return args


@contextlib.contextmanager
def exclusive_run(run_dir):
    """Prevent two launches from publishing checkpoints into the same run."""
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    with (Path(run_dir) / '.training.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another trainer is using {run_dir}') from exc
        yield


def log(message):
    if dist.get_rank() == 0:
        print(message, flush=True)


def checkpoints(run_dir):
    """Only fully published checkpoints; exclude interrupted .tmp directories."""
    return sorted((p for p in Path(run_dir).glob('checkpoint-*')
                   if p.name.removeprefix('checkpoint-').isdigit()
                   and (p / 'trainer_state.json').is_file() and (p / 'model/config.json').is_file()),
                  key=lambda p: int(p.name.rsplit('-', 1)[1]))


def model_directory(path):
    path = Path(path)
    if (path / 'model/config.json').is_file():
        path = path / 'model'
    with (path / 'config.json').open() as f:
        if json.load(f).get('model_type') != 'logkv':
            raise ValueError(f'{path} is not a LogKV model')
    return path


def resume_directory(spec, run_dir):
    if spec == 'latest':
        found = checkpoints(run_dir)
        if not found:
            raise FileNotFoundError(f'No completed checkpoints in {run_dir}')
        return found[-1]
    path = Path(spec)
    return path if path.is_dir() else Path(run_dir) / path


class EpochCursorSampler(Sampler):
    def __init__(self, dataset, seed):
        self.base = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True, seed=seed)
        self.skip = 0

    def set_cursor(self, epoch, samples):
        self.base.set_epoch(epoch)
        self.skip = samples

    def __iter__(self):
        return itertools.islice(iter(self.base), self.skip, None)

    def __len__(self):
        return max(0, len(self.base) - self.skip)


def make_optimizers(stage, lr):
    muon, adamw = split_params_for_muon(stage)
    optimizers = []
    if muon:
        optimizers.append(torch.optim.Muon(muon, lr=lr, adjust_lr_fn='match_rms_adamw'))
    if adamw:
        optimizers.append(torch.optim.AdamW(adamw, lr=lr, weight_decay=0.0))
    return optimizers


class PipelineEngine:
    def __init__(self, stage, device, n_microbatches, precision):
        self.stage, self.device = stage, device
        self.precision = precision
        self.denominator = 1
        pipe = PipelineStage(stage, dist.get_rank(), dist.get_world_size(), device)
        # Normalize once by the actual valid-token count across the whole
        # accumulation group. Schedule must NOT divide gradients again.
        self.schedule = Schedule1F1B(pipe, n_microbatches=n_microbatches,
                                    loss_fn=self.loss, scale_grads=False)

    def loss(self, logits, labels):
        # reduction=sum is finite even for an entirely ignored microbatch.
        return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                               labels.reshape(-1), ignore_index=-100,
                               reduction='sum') / self.denominator

    def backward(self, batches):
        self.denominator = max(1, sum(int((labels != -100).sum()) for _, labels in batches))
        loss = torch.zeros((), device=self.device)
        for ids, labels in batches:
            inputs = (ids.to(self.device, non_blocking=True),) if self.stage.is_first else ()
            targets = labels.to(self.device, non_blocking=True) if self.stage.is_last else None
            losses = []
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.precision == 'bf16'):
                self.schedule.step(*inputs, target=targets, losses=losses, return_outputs=False)
            if losses:
                loss += torch.stack([value.detach() for value in losses]).sum()
        dist.broadcast(loss, src=dist.get_world_size() - 1)
        return loss.item()


@torch.no_grad()
def clip_global_grad_norm(stage, max_norm, device):
    """One norm over disjoint parameters on ALL stages, not per-stage clipping."""
    grads = [p.grad for p in stage.parameters() if p.grad is not None]
    squared = torch.stack([g.float().norm().square() for g in grads]).sum() if grads else torch.zeros((), device=device)
    dist.all_reduce(squared)
    norm = squared.sqrt()
    if not torch.isfinite(norm):
        raise FloatingPointError(f'Nonfinite pipeline gradient norm: {norm.item()}')
    coefficient = (max_norm / (norm + 1e-6)).clamp(max=1.0)
    torch._foreach_mul_(grads, coefficient) if grads else None
    return norm.item()


def save_checkpoint(run_dir, stage, optimizers, tokenizer, trainer_state, keep):
    rank = dist.get_rank()
    target = Path(run_dir) / f"checkpoint-{trainer_state['step']}"
    # Reaching max-steps at an already saved boundary needs no second write.
    if target.is_dir():
        return target
    tmp = target.with_name(target.name + '.tmp')
    if rank == 0:
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
    dist.barrier()
    torch.save({'model': stage.state_dict(), 'stage_info': stage.stage_info}, tmp / f'stage_{rank}.pt')
    torch.save({'optimizers': [opt.state_dict() for opt in optimizers],
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
                'python_rng': random.getstate()}, tmp / f'optimizer_{rank}.pt')
    dist.barrier()
    if rank == 0:
        export_model(tmp, stage.config, tokenizer, trainer_state['stage_infos'])
        (tmp / 'trainer_state.json').write_text(json.dumps(trainer_state, indent=2) + '\n')
        os.replace(tmp, target)
        for old in checkpoints(run_dir)[:-keep]:
            shutil.rmtree(old)
        log(f'Saved {target}/model')
    dist.barrier()
    return target


def restore_training(stage, optimizers, checkpoint, expected):
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f'Resume mismatch for {key}; use --start-checkpoint for a weights-only restart')
    rank = dist.get_rank()
    weights = torch.load(checkpoint / f'stage_{rank}.pt', map_location='cpu', weights_only=True)
    if weights['stage_info'] != stage.stage_info:
        raise ValueError('Resume stage layout mismatch')
    stage.load_state_dict(weights['model'], strict=True)
    saved = torch.load(checkpoint / f'optimizer_{rank}.pt', map_location='cpu', weights_only=False)
    if len(saved['optimizers']) != len(optimizers):
        raise ValueError('Resume optimizer layout mismatch')
    for opt, value in zip(optimizers, saved['optimizers']):
        opt.load_state_dict(value)
    torch.set_rng_state(saved['torch_rng'])
    torch.cuda.set_rng_state(saved['cuda_rng'])
    random.setstate(saved['python_rng'])
    return state


def control_command(path, device):
    command = CMD_NONE
    if dist.get_rank() == 0:
        path = Path(path)
        try:
            command = _CMD_MAP.get(path.read_text().strip(), CMD_NONE)
            if command != CMD_NONE:
                path.unlink()
        except FileNotFoundError:
            pass
    value = torch.tensor(command, device=device)
    dist.broadcast(value, 0)
    return value.item()


@torch.no_grad()
def generate_samples(stage, tokenizer, device, step, path, max_new_tokens, precision):
    """Sequential inference across the same stages; no full GPU model needed."""
    rank, world = dist.get_rank(), dist.get_world_size()
    lines = [f'===== step {step} =====']
    training = stage.training
    stage.eval()
    try:
        with torch.random.fork_rng(devices=[device.index]), torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=precision == 'bf16'):
            torch.manual_seed(12345 + step)
            for prompt in SAMPLE_PROMPTS:
                ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
                tokens, hidden = ids[0].tolist(), None
                for t in range(max_new_tokens):
                    single = t > 0
                    shape = (1, stage.config.d_model) if single else (1, ids.size(1), stage.config.d_model)
                    x = ids[:, 0] if single else ids
                    if rank:
                        x = torch.empty(shape, device=device, dtype=next(stage.parameters()).dtype)
                        dist.recv(x, src=rank - 1)
                    out, hidden = stage.step(x, hidden, single_token=single)
                    token = torch.empty((1, 1), device=device, dtype=torch.long)
                    if not stage.is_last:
                        dist.send(out.contiguous(), dst=rank + 1)
                    else:
                        logits = out if single else out[:, -1]
                        probs, indices = logits.float().softmax(-1).sort(descending=True)
                        remove = probs.cumsum(-1) - probs >= 0.95
                        probs = probs.masked_fill(remove, 0)
                        token.copy_(indices.gather(-1, torch.multinomial(probs, 1)))
                    dist.broadcast(token, src=world - 1)
                    tokens.append(token.item())
                    ids = token
                    if token.item() == tokenizer.eos_token_id:
                        break
                if rank == 0:
                    lines.append(f'[{prompt}] {tokenizer.decode(tokens, skip_special_tokens=True)}')
    finally:
        stage.train(training)
    if rank == 0:
        message = '\n'.join(lines)
        print(message, flush=True)
        with Path(path).open('a') as f:
            f.write(message + '\n')


def prepare_data(args):
    """Serialize cache construction before NCCL initialization (shared filesystem)."""
    cache_dir = Path(get_data_dir()) / 'hf_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    # No stale sentinel after an interrupted build; the OS releases flock.
    with (cache_dir / '.logkv_pipeline_cache.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return prepare_all_datasets(args.context_length, cache_dir=str(cache_dir),
            prefault=not args.no_prefault and int(os.environ.get('RANK', 0)) == 0,
            dataset_type=args.dataset_type, num_workers=args.cache_build_workers)


def train(args, dataset, tokenizer):
    """Distributed group must already be initialized; also used by smoke tests."""
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', dist.get_rank())))
    run_dir = Path(get_data_dir()) / 'checkpoints_logkv_pipeline' / args.run_name
    with contextlib.ExitStack() as stack:
        error = [None]
        if dist.get_rank() == 0:
            try:
                stack.enter_context(exclusive_run(run_dir))
            except (OSError, RuntimeError) as exc:
                error[0] = str(exc)
        dist.broadcast_object_list(error, src=0)
        if error[0]:
            raise RuntimeError(error[0])
        return _train(args, dataset, tokenizer)


def _train(args, dataset, tokenizer):
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', rank)))
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    if args.n_microbatches < world:
        raise ValueError('1F1B requires n-microbatches >= number of stages')
    run_dir = Path(get_data_dir()) / 'checkpoints_logkv_pipeline' / args.run_name
    resume = resume_directory(args.resume, run_dir) if args.resume else None
    existing = checkpoints(run_dir)
    if existing:
        if resume is None:
            raise FileExistsError(f'{run_dir} already has checkpoints; use --resume or a new run-name')
        if resume.resolve() != existing[-1].resolve():
            raise ValueError('Output run already has other/later checkpoints; use a new run-name')
    source = model_directory(resume or args.start_checkpoint) if (resume or args.start_checkpoint) else None
    if source:
        config = LogKVConfig.from_pretrained(source, local_files_only=True)
        log(f'Architecture loaded from {source}; model architecture CLI flags are ignored')
    else:
        names = ['d_model', 'num_heads', 'd_ff', 'chunk_size', 'num_layers', 'phase_emb',
                 'phase_levels', 'learnable_decay', 'gated_attention', 'kv_norm',
                 'level_amplify', 'v_norm_only', 'self_slot', 'conv_kernel_size']
        config = LogKVConfig(**{key: getattr(args, key) for key in names},
            vocab_size=tokenizer.vocab_size, pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id)
    if (config.vocab_size, config.pad_token_id, config.bos_token_id, config.eos_token_id) != (
            tokenizer.vocab_size, tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id):
        raise ValueError('Dataset tokenizer does not match checkpoint vocabulary/special tokens')
    infos = LogKVLMPipelineStage.split_config(config.num_layers, world, args.stage_layer_split)
    stage = LogKVLMPipelineStage(config, **infos[rank]).to(device).train()
    optimizers = make_optimizers(stage, args.lr)
    signature = {key: getattr(args, key) for key in ['dataset_type', 'context_length',
        'batch_size', 'grad_accum', 'n_microbatches', 'seed', 'precision']}
    signature.update(stage_infos=infos, dataset_length=len(dataset))
    state = dict(signature, step=0, epoch=0, batch_in_epoch=0, ema_loss=None, args=vars(args))
    if resume:
        state = restore_training(stage, optimizers, resume, signature)
        state['args'] = vars(args)
    elif source:
        stage.load_model_directory(source)
    # Compare dataset sizes before starting communication with static shapes.
    lengths = [None] * world
    dist.all_gather_object(lengths, len(dataset))
    if len(set(lengths)) != 1:
        raise ValueError('Dataset size differs between stages')
    updates_per_epoch = len(dataset) // args.batch_size // args.grad_accum
    if updates_per_epoch == 0:
        raise ValueError('Dataset is smaller than one effective batch')
    usable_batches = updates_per_epoch * args.grad_accum
    if not 0 <= state['batch_in_epoch'] < usable_batches or state['batch_in_epoch'] % args.grad_accum:
        raise ValueError('Invalid checkpoint data cursor')
    total_params = torch.tensor(sum(p.numel() for p in stage.parameters()), device=device)
    dist.all_reduce(total_params)
    log(f'LogKV pipeline: {total_params.item():,} params; stages={world}; '
        f'layers={[i["layer_end"]-i["layer_start"] for i in infos]}; '
        f'effective batch={args.batch_size * args.grad_accum}; precision={args.precision}')
    sampler = EpochCursorSampler(dataset, args.seed)
    # Separate generator keeps iterator creation from perturbing model RNG on resume.
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        generator=torch.Generator().manual_seed(args.seed))
    engine = PipelineEngine(stage, device, args.n_microbatches, args.precision)
    writer = SummaryWriter(str(Path(get_data_dir()) / 'tensorboard' / 'logkv-pipeline' / args.run_name),
                           purge_step=state['step'] + 1 if resume else None) if rank == 0 else None
    run_dir.mkdir(parents=True, exist_ok=True)
    tick, logged_steps = time.monotonic(), 0
    try:
        while state['epoch'] < args.num_epochs and (not args.max_steps or state['step'] < args.max_steps):
            epoch = state['epoch']
            sampler.set_cursor(epoch, state['batch_in_epoch'] * args.batch_size)
            iterator = iter(loader)
            while state['epoch'] == epoch:
                cmd = control_command(args.control_file, device)
                if cmd == CMD_PAUSE:
                    log(f'Training paused; write resume or save_and_exit to {args.control_file}')
                while cmd == CMD_PAUSE:
                    time.sleep(1)
                    next_cmd = control_command(args.control_file, device)
                    if next_cmd in (CMD_RESUME, CMD_SAVE_AND_EXIT):
                        cmd = next_cmd
                if cmd == CMD_RESUME:
                    log('Training resumed')
                if cmd == CMD_SAVE_AND_EXIT:
                    return save_checkpoint(run_dir, stage, optimizers, tokenizer, state, args.max_checkpoints)
                batches = [next(iterator) for _ in range(args.grad_accum)]
                for ids, labels in batches:
                    if ids.shape != (args.batch_size, args.context_length - 1) or labels.shape != ids.shape:
                        raise ValueError('Dataset must yield context-length-1 input/label tensors')
                for opt in optimizers:
                    opt.zero_grad(set_to_none=True)
                loss = engine.backward(batches)
                grad_norm = clip_global_grad_norm(stage, args.grad_clip, device)
                lr = args.lr * min(1., (state['step'] + 1) / max(1, args.warmup))
                for opt in optimizers:
                    for group in opt.param_groups:
                        group['lr'] = lr
                    opt.step()
                state['step'] += 1
                state['batch_in_epoch'] += args.grad_accum
                if state['batch_in_epoch'] == usable_batches:
                    state['epoch'], state['batch_in_epoch'] = epoch + 1, 0
                state['ema_loss'] = loss if state['ema_loss'] is None else .99 * state['ema_loss'] + .01 * loss
                if writer:
                    for key, value in [('loss', loss), ('ema_loss', state['ema_loss']), ('grad_norm', grad_norm), ('lr', lr)]:
                        writer.add_scalar(f'train/{key}', value, state['step'])
                logged_steps += 1
                if state['step'] % args.log_interval == 0:
                    throughput = logged_steps * (args.context_length - 1) * args.batch_size * args.grad_accum / (time.monotonic() - tick)
                    log(f"epoch {epoch} step {state['step']} | loss {loss:.6f} | ema {state['ema_loss']:.6f} | "
                        f'lr {lr:.2e} | grad_norm {grad_norm:.4f} | {throughput:,.0f} tok/s')
                    tick, logged_steps = time.monotonic(), 0
                if args.sample_interval and state['step'] % args.sample_interval == 0:
                    generate_samples(stage, tokenizer, device, state['step'], run_dir / 'samples.log',
                                     args.sample_max_new_tokens, args.precision)
                if state['step'] % args.checkpoint_interval == 0:
                    save_checkpoint(run_dir, stage, optimizers, tokenizer, state, args.max_checkpoints)
                if args.max_steps and state['step'] >= args.max_steps:
                    break
        return save_checkpoint(run_dir, stage, optimizers, tokenizer, state, args.max_checkpoints)
    finally:
        if writer:
            writer.close()


def main():
    args = parse_args()
    dataset, tokenizer = prepare_data(args)
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    dist.init_process_group('nccl', timeout=timedelta(hours=8))
    try:
        train(args, dataset, tokenizer)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
