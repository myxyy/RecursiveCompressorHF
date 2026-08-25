"""
Data-parallel (DDP) training script for LogKVLM.

Usage:
    uv run torchrun --nproc_per_node=6 train_logkv.py
    uv run torchrun --nproc_per_node=6 train_logkv.py --resume latest

Control commands (write to control.cmd file during training):
    echo "pause"         > control.cmd   # Pause training
    echo "resume"        > control.cmd   # Resume training
    echo "save_and_exit" > control.cmd   # Save checkpoint and exit

Unlike train_pipeline.py (pipeline parallel for models too big for one GPU),
LogKVLM fits on a single GPU, so this uses plain data parallelism: every rank
holds the full model and sees a distinct shard of the data.

To check language acquisition during training, rank 0 periodically generates
continuations of fixed Japanese prompts (printed and appended to samples.log
in the checkpoint dir).
"""

import argparse
import os
import shutil
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from dotenv import load_dotenv

from configuration_logkv import LogKVConfig
from logkv_lm import LogKVLM
from dataset import prepare_all_datasets

load_dotenv()
torch.set_float32_matmul_precision("high")

CONTROL_FILE = "control.cmd"
CMD_NONE, CMD_PAUSE, CMD_RESUME, CMD_SAVE_AND_EXIT = 0, 1, 2, 3
_CMD_MAP = {"pause": CMD_PAUSE, "resume": CMD_RESUME, "save_and_exit": CMD_SAVE_AND_EXIT}

# Muon is for 2D hidden-layer weights; embedding and output head stay on AdamW
# per the original Muon recipe (biases/RMSNorms are 1D and fall through).
_ADAMW_ONLY_KEYWORDS = ("embedding", "head")

SAMPLE_PROMPTS = ["日本の首都は", "昔々あるところに", "人工知能とは"]


def parse_args():
    p = argparse.ArgumentParser(description="LogKVLM data-parallel training")
    p.add_argument("--run-name", type=str, default="base")
    p.add_argument("--dataset-type", type=str, default="pretrain",
                   choices=["pretrain", "instruct"])
    p.add_argument("--context-length", type=int, default=2048)
    p.add_argument("--d-model", type=int, default=1024)
    p.add_argument("--d-ff", type=int, default=3072)
    p.add_argument("--chunk-size", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=2, help="per-GPU micro batch")
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=1000, help="linear warmup steps")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=0, help="0 = full epoch(s)")
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--sample-interval", type=int, default=1000,
                   help="steps between Japanese-prompt sample generations (0 = off)")
    p.add_argument("--checkpoint-interval", type=int, default=1000)
    p.add_argument("--max-checkpoints", type=int, default=2)
    p.add_argument("--resume", type=str, default=None,
                   help="'latest' or a checkpoint dir under checkpoints_logkv/{run_name}")
    p.add_argument("--no-prefault", action="store_true",
                   help="skip memmap prefault (smoke tests)")
    return p.parse_args()


def get_data_dir():
    return os.environ.get("DATA_DIR", "./data")


def log(msg, rank=None):
    r = dist.get_rank() if rank is None else rank
    if r == 0:
        print(msg, flush=True)


def split_params_for_muon(model):
    muon_params, adamw_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        excluded = any(kw in name for kw in _ADAMW_ONLY_KEYWORDS)
        if param.ndim >= 2 and not excluded:
            muon_params.append(param)
        else:
            adamw_params.append(param)
    return muon_params, adamw_params


def read_control_command_synced(device):
    """Read control command on rank 0, broadcast to all ranks."""
    cmd = CMD_NONE
    if dist.get_rank() == 0 and os.path.exists(CONTROL_FILE):
        with open(CONTROL_FILE) as f:
            cmd = _CMD_MAP.get(f.read().strip(), CMD_NONE)
        if cmd != CMD_NONE and cmd != CMD_PAUSE:
            os.remove(CONTROL_FILE)
    t = torch.tensor([cmd], dtype=torch.long, device=device)
    dist.broadcast(t, src=0)
    return int(t.item())


def _list_checkpoints(run_dir):
    if not os.path.isdir(run_dir):
        return []
    names = [d for d in os.listdir(run_dir) if d.startswith("checkpoint-")]
    return sorted(names, key=lambda n: int(n.rsplit("-", 1)[1]))


def save_checkpoint(run_dir, step, epoch, model, optimizers, max_checkpoints):
    """rank 0 only. Saves HF model dir + trainer state for resume."""
    ckpt_dir = os.path.join(run_dir, f"checkpoint-{step}")
    tmp_dir = ckpt_dir + ".tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    model.save_pretrained(os.path.join(tmp_dir, "model"))
    torch.save({
        "step": step,
        "epoch": epoch,
        "optimizers_state_dict": [opt.state_dict() for opt in optimizers],
    }, os.path.join(tmp_dir, "trainer_state.pt"))
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
    os.replace(tmp_dir, ckpt_dir)
    for old in _list_checkpoints(run_dir)[:-max_checkpoints]:
        shutil.rmtree(os.path.join(run_dir, old))
    return ckpt_dir


@torch.no_grad()
def generate_samples(model, tokenizer, device, step, sample_path):
    """Fixed Japanese prompts -> continuations, for grammar-acquisition checks."""
    model.eval()
    lines = [f"===== step {step} ====="]
    for prompt in SAMPLE_PROMPTS:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model.generate(
                input_ids, max_new_tokens=80, do_sample=True,
                temperature=0.7, top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        lines.append(f"[{prompt}] {text}")
    model.train()
    msg = "\n".join(lines)
    print(msg, flush=True)
    with open(sample_path, "a") as f:
        f.write(msg + "\n")


def main():
    args = parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    data_dir = get_data_dir()
    cache_dir = os.path.join(data_dir, "hf_cache")
    run_dir = os.path.join(data_dir, "checkpoints_logkv", args.run_name)
    if rank == 0:
        os.makedirs(run_dir, exist_ok=True)

    # Dataset cache (rank 0 builds/prefaults, others wait then just open)
    sentinel_path = os.path.join(cache_dir, "mmap", f".cache_ready_logkv_{args.dataset_type}")
    if rank == 0:
        dataset, tokenizer = prepare_all_datasets(
            args.context_length, cache_dir=cache_dir,
            prefault=not args.no_prefault, dataset_type=args.dataset_type,
            num_workers=32,
        )
        with open(sentinel_path, "w") as f:
            f.write("ready")
    else:
        while not os.path.exists(sentinel_path):
            time.sleep(2)
        dataset, tokenizer = prepare_all_datasets(
            args.context_length, cache_dir=cache_dir,
            prefault=False, dataset_type=args.dataset_type, num_workers=1,
        )
    dist.barrier()
    if rank == 0 and os.path.exists(sentinel_path):
        os.remove(sentinel_path)

    config = LogKVConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        d_ff=args.d_ff,
        chunk_size=args.chunk_size,
        num_layers=args.num_layers,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Resume: load model weights before wrapping in DDP
    start_step, start_epoch = 0, 0
    resume_state = None
    if args.resume:
        name = args.resume
        if name == "latest":
            names = _list_checkpoints(run_dir)
            assert names, f"no checkpoints in {run_dir}"
            name = names[-1]
        ckpt_dir = name if os.path.isabs(name) else os.path.join(run_dir, name)
        log(f"Resuming from {ckpt_dir}")
        model = LogKVLM.from_pretrained(os.path.join(ckpt_dir, "model")).to(device)
        resume_state = torch.load(os.path.join(ckpt_dir, "trainer_state.pt"),
                                  map_location="cpu", weights_only=False)
        start_step = resume_state["step"]
        start_epoch = resume_state["epoch"]
    else:
        model = LogKVLM(config).to(device)
    model.train()
    num_params = sum(p.numel() for p in model.parameters())
    log(f"LogKVLM: {num_params:,} params | d_model={args.d_model} d_ff={args.d_ff} "
        f"layers={args.num_layers} chunk={args.chunk_size} ctx={args.context_length}")
    log(f"world={world_size} batch/GPU={args.batch_size} accum={args.grad_accum} "
        f"-> effective batch {world_size * args.batch_size * args.grad_accum}")

    ddp_model = DDP(model, device_ids=[local_rank])

    muon_params, adamw_params = split_params_for_muon(model)
    optimizers = []
    if muon_params:
        optimizers.append(torch.optim.Muon(
            muon_params, lr=args.lr, adjust_lr_fn="match_rms_adamw"))
    optimizers.append(torch.optim.AdamW(adamw_params, lr=args.lr, weight_decay=0.0))
    if resume_state is not None:
        for opt, st in zip(optimizers, resume_state["optimizers_state_dict"]):
            opt.load_state_dict(st)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=True, seed=args.seed, drop_last=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        num_workers=2, pin_memory=True, drop_last=True)

    writer = SummaryWriter(
        os.path.join(data_dir, "tensorboard", f"logkv-{args.dataset_type}",
                     args.run_name)) if rank == 0 else None
    sample_path = os.path.join(run_dir, "samples.log")

    step = start_step
    ema_loss = None
    paused = False
    stop = False
    t0 = time.time()
    for epoch in range(start_epoch, args.num_epochs):
        sampler.set_epoch(epoch)
        micro_iter = iter(loader)
        # NOTE: resume replays the epoch's shuffled order from its start (the
        # sampler order is reproducible via seed+set_epoch, but we do not skip
        # already-seen batches; acceptable for these experiments).
        while True:
            cmd = read_control_command_synced(device)
            if cmd == CMD_PAUSE and not paused:
                paused = True
                log("Training paused. Write 'resume' to control.cmd to continue.")
            elif cmd == CMD_RESUME and paused:
                paused = False
                log("Training resumed.")
            elif cmd == CMD_SAVE_AND_EXIT:
                stop = True
            if paused and not stop:
                time.sleep(2)
                continue
            if stop:
                break

            # ---- one optimizer step (grad_accum micro-batches) ----
            micros = []
            try:
                for _ in range(args.grad_accum):
                    micros.append(next(micro_iter))
            except StopIteration:
                break  # epoch exhausted (drop incomplete accumulation group)

            loss_sum = 0.0
            for i, (input_ids, labels) in enumerate(micros):
                input_ids = input_ids.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                ctx = (ddp_model.no_sync() if i < len(micros) - 1
                       else torch.enable_grad())
                with ctx:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out = ddp_model(input_ids, labels=labels)
                    (out.loss / args.grad_accum).backward()
                loss_sum += out.loss.item() / args.grad_accum

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            lr = args.lr * min(1.0, (step + 1) / max(1, args.warmup))
            for opt in optimizers:
                for g in opt.param_groups:
                    g["lr"] = lr
                opt.step()
                opt.zero_grad()
            step += 1

            ema_loss = loss_sum if ema_loss is None else 0.99 * ema_loss + 0.01 * loss_sum
            if writer:
                writer.add_scalar("train/loss", loss_sum, step)
                writer.add_scalar("train/grad_norm", grad_norm.item(), step)
                writer.add_scalar("train/lr", lr, step)

            if step % args.log_interval == 0:
                elapsed = time.time() - t0
                tokens_per_sec = (args.context_length * args.batch_size *
                                  args.grad_accum * world_size * args.log_interval) / max(
                                      1e-9, elapsed)
                log(f"epoch {epoch} step {step} | loss {loss_sum:.4f} | "
                    f"ema {ema_loss:.4f} | lr {lr:.2e} | "
                    f"grad_norm {grad_norm.item():.2f} | {tokens_per_sec:,.0f} tok/s")
                t0 = time.time()

            if args.sample_interval and step % args.sample_interval == 0 and rank == 0:
                generate_samples(model, tokenizer, device, step, sample_path)
            if args.sample_interval and step % args.sample_interval == 0:
                dist.barrier()  # keep ranks in lockstep around generation

            if step % args.checkpoint_interval == 0:
                if rank == 0:
                    ckpt = save_checkpoint(run_dir, step, epoch, model, optimizers,
                                           args.max_checkpoints)
                    log(f"Saved {ckpt}")
                dist.barrier()

            if args.max_steps and step - start_step >= args.max_steps:
                stop = True
                break
        if stop:
            break

    if rank == 0:
        ckpt = save_checkpoint(run_dir, step, epoch, model, optimizers,
                               args.max_checkpoints)
        log(f"Final checkpoint: {ckpt}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
