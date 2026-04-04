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
CONTEXT_LENGTH = 4096
LEARNING_RATE = 3e-4
NUM_EPOCHS = 1
GRAD_CLIP = 1.0
CHECKPOINT_INTERVAL = 1000  # Save every N steps
MAX_CHECKPOINTS = 2
VALIDATION_RATIO = 0.001
CONTROL_FILE = "control.cmd"


def get_data_dir():
    return os.environ.get("DATA_DIR", "./data")


def get_checkpoint_dir():
    return os.path.join(get_data_dir(), "checkpoints")


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def log(msg):
    if is_main_process():
        print(msg, flush=True)


def read_control_command():
    """Read and clear the control file."""
    if not os.path.exists(CONTROL_FILE):
        return None
    try:
        with open(CONTROL_FILE, "r") as f:
            cmd = f.read().strip()
        os.remove(CONTROL_FILE)
        return cmd
    except (OSError, IOError):
        return None


def save_checkpoint(model, optimizer, step, epoch, checkpoint_dir):
    """Save checkpoint, keeping only the latest MAX_CHECKPOINTS."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint-{step}")

    # Save model (unwrap DDP if needed)
    unwrapped = model.module if isinstance(model, DDP) else model
    unwrapped.save_pretrained(path)

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

    # Load model weights
    unwrapped = model.module if isinstance(model, DDP) else model
    state_dict = torch.load(os.path.join(latest, "model.safetensors"), map_location="cpu", weights_only=True)
    unwrapped.load_state_dict(state_dict)

    # Load training state
    training_state = torch.load(os.path.join(latest, "training_state.pt"), map_location="cpu", weights_only=True)
    optimizer.load_state_dict(training_state["optimizer_state_dict"])

    return training_state["step"], training_state["epoch"]


def train():
    # Determine rank before init_process_group (torchrun sets these env vars)
    distributed = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))

    data_dir = get_data_dir()
    checkpoint_dir = get_checkpoint_dir()
    cache_dir = os.path.join(data_dir, "hf_cache")

    # Build dataset cache BEFORE init_process_group to avoid NCCL timeout.
    # Rank 0 builds, others poll for completion via sentinel file.
    sentinel_path = os.path.join(cache_dir, "mmap", ".cache_ready")
    if rank == 0:
        print("Preparing datasets...", flush=True)
        prepare_all_datasets(CONTEXT_LENGTH, cache_dir=cache_dir)
        os.makedirs(os.path.dirname(sentinel_path), exist_ok=True)
        with open(sentinel_path, "w") as f:
            f.write("ready")
        print("Cache ready.", flush=True)
    else:
        while not os.path.exists(sentinel_path):
            time.sleep(5)

    # Now safe to init process group — all ranks have cache ready
    if distributed:
        dist.init_process_group("nccl")
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size_per_gpu = 1

    # Tokenizer and config
    tokenizer = get_tokenizer()
    config = RecursiveCompressorConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=1024,
        num_heads=8,
        d_ff=2048,
        chunk_size=8,
        compress_size=4,
        num_layers=8,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Load cached datasets (all ranks — fast, just opens memmaps)
    full_dataset, _ = prepare_all_datasets(CONTEXT_LENGTH, cache_dir=cache_dir)
    # Remove sentinel so next run rebuilds if needed
    if rank == 0 and os.path.exists(sentinel_path):
        os.remove(sentinel_path)

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

    # Model
    model = RecursiveCompressorLM(config).to(device)
    if distributed:
        model = DDP(model, device_ids=[rank])

    num_params = sum(p.numel() for p in model.parameters())
    log(f"Parameters: {num_params:,}")
    log(f"Device: {device}, World size: {world_size}")

    # Optimizer
    optimizer = RAdamScheduleFree(model.parameters(), lr=LEARNING_RATE)

    # Resume from checkpoint
    start_step, start_epoch = load_latest_checkpoint(model, optimizer, checkpoint_dir)
    global_step = start_step

    # Training loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        optimizer.train()
        if distributed:
            train_sampler.set_epoch(epoch)

        total_loss = 0.0
        num_batches = 0
        paused = False

        for batch_idx, (input_ids, labels) in enumerate(train_loader):
            # Skip already-processed batches on resume
            if epoch == start_epoch and batch_idx < start_step:
                continue

            # Check control commands
            cmd = read_control_command()
            if cmd == "pause":
                log("Training paused. Write 'resume' to control.cmd to continue.")
                paused = True
            elif cmd == "resume":
                log("Training resumed.")
                paused = False
            elif cmd == "save_and_exit":
                log("Save and exit requested.")
                if is_main_process():
                    save_checkpoint(model, optimizer, global_step, epoch, checkpoint_dir)
                if distributed:
                    dist.barrier()
                    dist.destroy_process_group()
                return

            while paused:
                time.sleep(1)
                cmd = read_control_command()
                if cmd == "resume":
                    log("Training resumed.")
                    paused = False
                elif cmd == "save_and_exit":
                    log("Save and exit requested.")
                    if is_main_process():
                        save_checkpoint(model, optimizer, global_step, epoch, checkpoint_dir)
                    if distributed:
                        dist.barrier()
                        dist.destroy_process_group()
                    return

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            output = model(input_ids, labels=labels)
            loss = output.loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            global_step += 1

            if global_step % 100 == 0:
                avg_loss = total_loss / num_batches
                log(f"Epoch {epoch+1}/{NUM_EPOCHS}, Step {global_step}, Loss: {avg_loss:.4f}")

            if global_step % CHECKPOINT_INTERVAL == 0 and is_main_process():
                save_checkpoint(model, optimizer, global_step, epoch, checkpoint_dir)
                if distributed:
                    dist.barrier()

        # Epoch-end validation
        model.eval()
        optimizer.eval()
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

        # Epoch-end checkpoint
        if is_main_process():
            save_checkpoint(model, optimizer, global_step, epoch + 1, checkpoint_dir)
        if distributed:
            dist.barrier()

    # Save final model
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
