"""
DDP training script for RecursiveCompressorLM.

Usage:
    # Single GPU
    uv run python train.py

    # Multi-GPU (e.g. 6 GPUs)
    uv run torchrun --nproc_per_node=6 train.py

Control commands (write to control.cmd file during training):
    echo "pause"         > control.cmd   # Pause training
    echo "resume"        > control.cmd   # Resume training
    echo "save_and_exit" > control.cmd   # Save checkpoint and exit
"""

import os
import time
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from dotenv import load_dotenv
from schedulefree import RAdamScheduleFree

from configuration_recursive_compressor import RecursiveCompressorConfig
from recursive_compressor_lm import RecursiveCompressorLM
from dataset import prepare_all_datasets, get_tokenizer

load_dotenv()

# Hyperparameters
CONTEXT_LENGTH = 2048
LEARNING_RATE = 3e-4
NUM_EPOCHS = 1
GRAD_CLIP = 1.0
GRAD_ACCUM_STEPS = 4
CHECKPOINT_INTERVAL = 1000  # Save every N optimizer steps
MAX_CHECKPOINTS = 2
VALIDATION_RATIO = 0.001
CONTROL_FILE = "control.cmd"
LOG_INTERVAL = 10  # Log every N optimizer steps


def get_data_dir():
    return os.environ.get("DATA_DIR", "./data")


def get_checkpoint_dir():
    return os.path.join(get_data_dir(), "checkpoints")


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def log(msg):
    if is_main_process():
        print(msg, flush=True)


CMD_NONE = 0
CMD_PAUSE = 1
CMD_RESUME = 2
CMD_SAVE_AND_EXIT = 3
_CMD_MAP = {"pause": CMD_PAUSE, "resume": CMD_RESUME, "save_and_exit": CMD_SAVE_AND_EXIT}


def read_control_command_synced(device, distributed):
    """Read control command on rank 0, broadcast to all ranks.
    Returns one of CMD_NONE, CMD_PAUSE, CMD_RESUME, CMD_SAVE_AND_EXIT."""
    cmd_tensor = torch.zeros(1, dtype=torch.long, device=device)

    if not distributed or dist.get_rank() == 0:
        if os.path.exists(CONTROL_FILE):
            try:
                with open(CONTROL_FILE, "r") as f:
                    cmd_str = f.read().strip()
                os.remove(CONTROL_FILE)
                cmd_tensor[0] = _CMD_MAP.get(cmd_str, CMD_NONE)
            except (OSError, IOError):
                pass

    if distributed:
        dist.broadcast(cmd_tensor, src=0)

    return cmd_tensor.item()


def save_checkpoint(model, optimizer, step, epoch, checkpoint_dir, tokenizer=None):
    """Save checkpoint, keeping only the latest MAX_CHECKPOINTS."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint-{step}")

    # Save model (unwrap DDP if needed)
    unwrapped = model.module if isinstance(model, DDP) else model
    unwrapped.save_pretrained(path)

    # Save tokenizer alongside model for predict.py compatibility
    if tokenizer is not None:
        tokenizer.save_pretrained(path)

    # Save optimizer and training state
    torch.save({
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
    }, os.path.join(path, "training_state.pt"))

    log(f"Checkpoint saved: {path}")

    # Remove old checkpoints
    checkpoints = sorted(
        [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1]),
    )
    while len(checkpoints) > MAX_CHECKPOINTS:
        old = os.path.join(checkpoint_dir, checkpoints.pop(0))
        import shutil
        shutil.rmtree(old)
        log(f"Removed old checkpoint: {old}")


def load_latest_checkpoint(model, optimizer, checkpoint_dir):
    """Load the latest checkpoint if available. Returns (step, epoch)."""
    if not os.path.exists(checkpoint_dir):
        return 0, 0

    checkpoints = sorted(
        [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1]),
    )
    if not checkpoints:
        return 0, 0

    latest = os.path.join(checkpoint_dir, checkpoints[-1])
    log(f"Resuming from checkpoint: {latest}")

    # Load model weights via from_pretrained (reads safetensors correctly)
    unwrapped = model.module if isinstance(model, DDP) else model
    loaded = RecursiveCompressorLM.from_pretrained(latest, torch_dtype=torch.bfloat16)
    unwrapped.load_state_dict(loaded.state_dict())
    del loaded

    # Load training state (weights_only=False needed for schedule_free optimizer state)
    training_state = torch.load(os.path.join(latest, "training_state.pt"), map_location="cpu", weights_only=False)
    optimizer.load_state_dict(training_state["optimizer_state_dict"])

    return training_state["step"], training_state["epoch"]


def train():
    # Determine rank before init_process_group (torchrun sets these env vars)
    distributed = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    data_dir = get_data_dir()
    checkpoint_dir = get_checkpoint_dir()
    cache_dir = os.path.join(data_dir, "hf_cache")

    # Build dataset cache BEFORE init_process_group to avoid NCCL timeout.
    # Rank 0 builds, others poll for completion via sentinel file.
    sentinel_path = os.path.join(cache_dir, "mmap", ".cache_ready")
    if rank == 0:
        print("Preparing datasets...", flush=True)
        full_dataset, tokenizer = prepare_all_datasets(CONTEXT_LENGTH, cache_dir=cache_dir)
        with open(sentinel_path, "w") as f:
            f.write("ready")
        print("Cache ready.", flush=True)
    else:
        print(f"[rank {rank}] Waiting for cache...", flush=True)
        while not os.path.exists(sentinel_path):
            time.sleep(2)
        print(f"[rank {rank}] Loading cached datasets...", flush=True)
        full_dataset, tokenizer = prepare_all_datasets(CONTEXT_LENGTH, cache_dir=cache_dir)
        print(f"[rank {rank}] Datasets loaded.", flush=True)

    # All ranks have datasets loaded — now safe to init process group
    print(f"[rank {rank}] Initializing process group...", flush=True)
    if distributed:
        dist.init_process_group("nccl")
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Remove sentinel after all ranks have initialized
    if distributed:
        dist.barrier()
    if rank == 0 and os.path.exists(sentinel_path):
        os.remove(sentinel_path)

    batch_size_per_gpu = 1
    effective_batch_size = batch_size_per_gpu * world_size * GRAD_ACCUM_STEPS

    config = RecursiveCompressorConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=2048,
        num_heads=16,
        d_ff=4096,
        chunk_size=8,
        compress_size=4,
        num_layers=32,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    val_size = max(1, int(len(full_dataset) * VALIDATION_RATIO))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    log(f"Train: {train_size}, Validation: {val_size}")

    # DataLoader
    if distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size_per_gpu,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size_per_gpu,
        sampler=val_sampler, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # Model (bfloat16)
    model = RecursiveCompressorLM(config).to(dtype=torch.bfloat16, device=device)
    if distributed:
        model = DDP(model, device_ids=[local_rank])

    num_params = sum(p.numel() for p in model.parameters())
    log(f"Parameters: {num_params:,}")
    log(f"Device: {device}, World size: {world_size}, Effective batch size: {effective_batch_size}")
    log(f"Context length: {CONTEXT_LENGTH}, Grad accum steps: {GRAD_ACCUM_STEPS}, dtype: bfloat16")

    # Optimizer
    optimizer = RAdamScheduleFree(model.parameters(), lr=LEARNING_RATE)

    # Resume from checkpoint
    start_step, start_epoch = load_latest_checkpoint(model, optimizer, checkpoint_dir)
    global_step = start_step

    # Training loop
    total_optimizer_steps = len(train_loader) // GRAD_ACCUM_STEPS
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        optimizer.train()
        if distributed:
            train_sampler.set_epoch(epoch)

        accum_loss = 0.0
        ema_loss = None
        EMA_BETA = 0.99
        num_optimizer_steps = 0
        paused = False
        skip_batches = start_step * GRAD_ACCUM_STEPS if epoch == start_epoch else 0
        if skip_batches > 0:
            log(f"Skipping {skip_batches} batches to resume...")
        train_start_time = time.time()
        step_start_time = time.time()

        for batch_idx, (input_ids, labels) in enumerate(train_loader):
            if batch_idx < skip_batches:
                continue

            micro_step = batch_idx % GRAD_ACCUM_STEPS

            # Check control commands at the start of each accumulation cycle
            if micro_step == 0:
                cmd = read_control_command_synced(device, distributed)
                if cmd == CMD_PAUSE:
                    log("Training paused. Write 'resume' to control.cmd to continue.")
                    paused = True
                elif cmd == CMD_RESUME:
                    paused = False
                elif cmd == CMD_SAVE_AND_EXIT:
                    log("Save and exit requested.")
                    if is_main_process():
                        save_checkpoint(model, optimizer, global_step, epoch, checkpoint_dir, tokenizer)
                    if distributed:
                        dist.barrier()
                        dist.destroy_process_group()
                    return

                while paused:
                    time.sleep(1)
                    cmd = read_control_command_synced(device, distributed)
                    if cmd == CMD_RESUME:
                        log("Training resumed.")
                        paused = False
                    elif cmd == CMD_SAVE_AND_EXIT:
                        log("Save and exit requested.")
                        if is_main_process():
                            save_checkpoint(model, optimizer, global_step, epoch, checkpoint_dir, tokenizer)
                        if distributed:
                            dist.barrier()
                            dist.destroy_process_group()
                        return

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # Disable gradient sync for accumulation steps (except the last)
            context = model.no_sync() if (distributed and micro_step < GRAD_ACCUM_STEPS - 1) else nullcontext()
            with context:
                output = model(input_ids, labels=labels)
                loss = output.loss / GRAD_ACCUM_STEPS
                loss.backward()

            accum_loss += output.loss.item()

            # Optimizer step at the end of accumulation
            if micro_step == GRAD_ACCUM_STEPS - 1:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP).item()
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                num_optimizer_steps += 1
                step_loss = accum_loss / GRAD_ACCUM_STEPS
                ema_loss = step_loss if ema_loss is None else EMA_BETA * ema_loss + (1 - EMA_BETA) * step_loss

                if global_step % LOG_INTERVAL == 0:
                    elapsed = time.time() - train_start_time
                    step_time = time.time() - step_start_time
                    tokens_per_sec = (CONTEXT_LENGTH * batch_size_per_gpu * world_size * GRAD_ACCUM_STEPS * LOG_INTERVAL) / step_time
                    remaining_steps = total_optimizer_steps - (start_step + num_optimizer_steps)
                    if num_optimizer_steps > 0:
                        secs_per_step = elapsed / num_optimizer_steps
                        eta_secs = secs_per_step * remaining_steps
                        eta_h = int(eta_secs // 3600)
                        eta_m = int((eta_secs % 3600) // 60)
                        eta_str = f"{eta_h}h{eta_m:02d}m"
                    else:
                        eta_str = "..."
                    log(
                        f"Step {global_step}/{total_optimizer_steps} | "
                        f"Loss: {ema_loss:.4f} | "
                        f"GradNorm: {grad_norm:.4f} | "
                        f"Tok/s: {tokens_per_sec:.0f} | "
                        f"ETA: {eta_str}"
                    )
                    step_start_time = time.time()

                if global_step % CHECKPOINT_INTERVAL == 0:
                    if is_main_process():
                        save_checkpoint(model, optimizer, global_step, epoch, checkpoint_dir, tokenizer)
                    if distributed:
                        dist.barrier()

                accum_loss = 0.0

        # Epoch-end checkpoint (save BEFORE optimizer.eval() to keep training params)
        if is_main_process():
            save_checkpoint(model, optimizer, global_step, epoch + 1, checkpoint_dir, tokenizer)
        if distributed:
            dist.barrier()

        # Epoch-end validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                output = model(input_ids, labels=labels)
                val_loss += output.loss.item()
                val_batches += 1

        if distributed:
            val_tensor = torch.tensor([val_loss, val_batches], device=device)
            dist.all_reduce(val_tensor)
            val_loss, val_batches = val_tensor.tolist()

        if val_batches > 0:
            log(f"Epoch {epoch+1}/{NUM_EPOCHS}, Validation Loss: {val_loss / val_batches:.4f}")

    # Save final model (training params, not averaged)
    if is_main_process():
        final_path = os.path.join(data_dir, "final_model")
        unwrapped = model.module if isinstance(model, DDP) else model
        unwrapped.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        log(f"Final model saved: {final_path}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    train()
