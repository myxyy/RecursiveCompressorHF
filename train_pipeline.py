"""
Pipeline parallel training script for RecursiveCompressorLM.

Usage:
    uv run torchrun --nproc_per_node=6 train_pipeline.py

Control commands (write to control.cmd file during training):
    echo "pause"         > control.cmd   # Pause training
    echo "resume"        > control.cmd   # Resume training
    echo "save_and_exit" > control.cmd   # Save checkpoint and exit
"""

import os
import shutil
import time
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import ScheduleGPipe
from dotenv import load_dotenv
from schedulefree import RAdamScheduleFree

from configuration_recursive_compressor import RecursiveCompressorConfig
from recursive_compressor_lm import RecursiveCompressorLM
from recursive_compressor_lm_pipeline import RecursiveCompressorLMPipelineStage
from dataset import prepare_all_datasets, get_tokenizer

load_dotenv()

# Hyperparameters
CONTEXT_LENGTH = 2048
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1
GRAD_CLIP = 1.0
N_MICROBATCHES = 6
BATCH_SIZE = 6  # Must be >= N_MICROBATCHES
CHECKPOINT_INTERVAL = 1000
MAX_CHECKPOINTS = 2
VALIDATION_RATIO = 0.001
CONTROL_FILE = "control.cmd"
LOG_INTERVAL = 10
DATASET_IN_MEMORY = True  # Load entire memmap caches into RAM (faster, high memory use)


def get_data_dir():
    return os.environ.get("DATA_DIR", "./data")


def get_checkpoint_dir():
    return os.path.join(get_data_dir(), "checkpoints_pipeline")


CMD_NONE = 0
CMD_PAUSE = 1
CMD_RESUME = 2
CMD_SAVE_AND_EXIT = 3
_CMD_MAP = {"pause": CMD_PAUSE, "resume": CMD_RESUME, "save_and_exit": CMD_SAVE_AND_EXIT}


def read_control_command_synced(device):
    """Read control command on rank 0, broadcast to all ranks."""
    cmd_tensor = torch.zeros(1, dtype=torch.long, device=device)
    if dist.get_rank() == 0:
        if os.path.exists(CONTROL_FILE):
            try:
                with open(CONTROL_FILE, "r") as f:
                    cmd_str = f.read().strip()
                os.remove(CONTROL_FILE)
                cmd_tensor[0] = _CMD_MAP.get(cmd_str, CMD_NONE)
            except (OSError, IOError):
                pass
    dist.broadcast(cmd_tensor, src=0)
    return cmd_tensor.item()


def log(msg):
    if dist.get_rank() == 0:
        print(msg, flush=True)


def save_stage_checkpoint(stage_module, optimizer, step, epoch, checkpoint_dir,
                          rank, stage_info, tokenizer=None, config=None):
    """Save per-stage checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint-{step}")
    os.makedirs(path, exist_ok=True)

    torch.save({
        "stage_state_dict": stage_module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "stage_info": stage_info,
        "rank": rank,
    }, os.path.join(path, f"stage_{rank}.pt"))

    if rank == 0:
        if tokenizer is not None:
            tokenizer.save_pretrained(path)
        if config is not None:
            config.save_pretrained(path)

    dist.barrier()

    # Rank 0 saves full model and removes old checkpoints
    if rank == 0:
        save_full_model(path, checkpoint_dir, step)
        _rotate_checkpoints(checkpoint_dir)


def save_full_model(checkpoint_path, checkpoint_dir, step):
    """Gather all stage state dicts and save as full model."""
    world_size = dist.get_world_size()
    gathered = []
    for r in range(world_size):
        stage_path = os.path.join(checkpoint_path, f"stage_{r}.pt")
        data = torch.load(stage_path, map_location="cpu", weights_only=False)
        gathered.append((data["rank"], data["stage_info"], data["stage_state_dict"]))

    full_state = RecursiveCompressorLMPipelineStage.reconstruct_full_state_dict(gathered)
    torch.save(full_state, os.path.join(checkpoint_path, "full_model.pt"))


def _rotate_checkpoints(checkpoint_dir):
    """Keep only the latest MAX_CHECKPOINTS."""
    checkpoints = sorted(
        [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1]),
    )
    while len(checkpoints) > MAX_CHECKPOINTS:
        old = os.path.join(checkpoint_dir, checkpoints.pop(0))
        shutil.rmtree(old)
        log(f"Removed old checkpoint: {old}")


def load_latest_checkpoint(stage_module, optimizer, checkpoint_dir, rank):
    """Load latest per-stage checkpoint. Returns (step, epoch)."""
    if not os.path.exists(checkpoint_dir):
        return 0, 0

    checkpoints = sorted(
        [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1]),
    )
    if not checkpoints:
        return 0, 0

    latest = os.path.join(checkpoint_dir, checkpoints[-1])
    stage_path = os.path.join(latest, f"stage_{rank}.pt")

    if os.path.exists(stage_path):
        log(f"Resuming from checkpoint: {latest}")
        data = torch.load(stage_path, map_location="cpu", weights_only=False)
        stage_module.load_state_dict(data["stage_state_dict"])
        optimizer.load_state_dict(data["optimizer_state_dict"])
        return data["step"], data["epoch"]

    # Fall back to full model
    full_path = os.path.join(latest, "full_model.pt")
    if os.path.exists(full_path):
        log(f"Resuming from full model: {full_path}")
        full_state = torch.load(full_path, map_location="cpu", weights_only=False)
        stage_module.load_from_full_model(full_state)
        return 0, 0

    return 0, 0


def train():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    data_dir = get_data_dir()
    checkpoint_dir = get_checkpoint_dir()
    cache_dir = os.path.join(data_dir, "hf_cache")

    # Dataset cache (rank 0 builds, others wait)
    sentinel_path = os.path.join(cache_dir, "mmap", ".cache_ready")
    if rank == 0:
        print("Preparing datasets...", flush=True)
        full_dataset, tokenizer = prepare_all_datasets(CONTEXT_LENGTH, cache_dir=cache_dir, in_memory=DATASET_IN_MEMORY)
        with open(sentinel_path, "w") as f:
            f.write("ready")
        print("Cache ready.", flush=True)
    else:
        while not os.path.exists(sentinel_path):
            time.sleep(2)
        full_dataset, tokenizer = prepare_all_datasets(CONTEXT_LENGTH, cache_dir=cache_dir, in_memory=DATASET_IN_MEMORY)

    dist.barrier()
    if rank == 0 and os.path.exists(sentinel_path):
        os.remove(sentinel_path)

    config = RecursiveCompressorConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=2048,
        num_heads=16,
        d_ff=4096,
        chunk_size=4,
        compress_size=1,
        num_layers=16,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Split model across pipeline stages
    stage_infos = RecursiveCompressorLMPipelineStage.split_config(config.num_layers, world_size)
    stage_info = stage_infos[rank]
    stage_module = RecursiveCompressorLMPipelineStage(config, **stage_info).to(device)

    stage_params = sum(p.numel() for p in stage_module.parameters())
    total_params_tensor = torch.tensor([stage_params], dtype=torch.long, device=device)
    dist.all_reduce(total_params_tensor)
    log(f"Total layers: {config.num_layers}, Stages: {world_size}, Total params: {total_params_tensor.item():,}")
    log(f"Stage {rank}: layers {stage_info['layer_start']}-{stage_info['layer_end']}, "
        f"params: {stage_params:,}, first={stage_info['is_first']}, last={stage_info['is_last']}")

    # Data (all ranks see the same data — pipeline parallel, not data parallel)
    val_size = max(1, int(len(full_dataset) * VALIDATION_RATIO))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    log(f"Train: {train_size}, Validation: {val_size}")

    train_sampler = DistributedSampler(train_dataset, num_replicas=1, rank=0, shuffle=True)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        sampler=train_sampler, num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=True, drop_last=True,
    )

    # Optimizer
    optimizer = RAdamScheduleFree(stage_module.parameters(), lr=LEARNING_RATE)

    # Resume
    start_step, start_epoch = load_latest_checkpoint(stage_module, optimizer, checkpoint_dir, rank)
    global_step = start_step

    # Loss function (only used on last stage)
    vocab_size = config.vocab_size

    def loss_fn(logits, targets):
        return nn.CrossEntropyLoss()(logits.float().view(-1, vocab_size), targets.view(-1))

    # Training loop
    total_steps = len(train_loader)
    for epoch in range(start_epoch, NUM_EPOCHS):
        stage_module.train()
        optimizer.train()
        train_sampler.set_epoch(epoch)

        ema_loss = None
        EMA_BETA = 0.99
        num_steps = 0
        paused = False
        skip_batches = start_step if epoch == start_epoch else 0
        if skip_batches > 0:
            log(f"Skipping {skip_batches} batches to resume...")
        train_start_time = time.time()
        step_start_time = time.time()

        for batch_idx, (input_ids, labels) in enumerate(train_loader):
            if batch_idx < skip_batches:
                continue

            # Control commands
            cmd = read_control_command_synced(device)
            if cmd == CMD_PAUSE:
                log("Training paused. Write 'resume' to control.cmd to continue.")
                paused = True
            elif cmd == CMD_RESUME:
                paused = False
            elif cmd == CMD_SAVE_AND_EXIT:
                log("Save and exit requested.")
                optimizer.eval()
                save_stage_checkpoint(stage_module, optimizer, global_step, epoch,
                                      checkpoint_dir, rank, stage_info, tokenizer, config)
                dist.destroy_process_group()
                return

            while paused:
                time.sleep(1)
                cmd = read_control_command_synced(device)
                if cmd == CMD_RESUME:
                    log("Training resumed.")
                    paused = False
                elif cmd == CMD_SAVE_AND_EXIT:
                    log("Save and exit requested.")
                    optimizer.eval()
                    save_stage_checkpoint(stage_module, optimizer, global_step, epoch,
                                          checkpoint_dir, rank, stage_info, tokenizer, config)
                    dist.destroy_process_group()
                    return

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # Build pipeline schedule for this step
            pipe_stage = PipelineStage(
                stage_module, stage_index=rank, num_stages=world_size, device=device,
            )
            schedule = ScheduleGPipe(pipe_stage, n_microbatches=N_MICROBATCHES, loss_fn=loss_fn)

            # Execute pipeline (losses are collected via the losses list arg)
            microbatch_losses = []
            if rank == 0:
                schedule.step(input_ids)
            elif rank == world_size - 1:
                schedule.step(target=labels, losses=microbatch_losses)
            else:
                schedule.step()

            # Gradient clipping and optimizer step
            grad_norm = nn.utils.clip_grad_norm_(stage_module.parameters(), GRAD_CLIP).item()
            optimizer.step()
            optimizer.zero_grad()

            # Loss tracking (last rank collects microbatch losses, broadcasts to rank 0)
            loss_tensor = torch.zeros(1, device=device)
            if rank == world_size - 1 and microbatch_losses:
                loss_tensor[0] = sum(l.item() for l in microbatch_losses) / len(microbatch_losses)
            dist.broadcast(loss_tensor, src=world_size - 1)
            batch_loss = loss_tensor.item()

            ema_loss = batch_loss if ema_loss is None else EMA_BETA * ema_loss + (1 - EMA_BETA) * batch_loss
            global_step += 1
            num_steps += 1

            if global_step % LOG_INTERVAL == 0:
                elapsed = time.time() - train_start_time
                step_time = time.time() - step_start_time
                tokens_per_sec = (CONTEXT_LENGTH * BATCH_SIZE * LOG_INTERVAL) / step_time
                remaining_steps = total_steps - (skip_batches + num_steps)
                if num_steps > 0:
                    secs_per_step = elapsed / num_steps
                    eta_secs = secs_per_step * remaining_steps
                    eta_h = int(eta_secs // 3600)
                    eta_m = int((eta_secs % 3600) // 60)
                    eta_str = f"{eta_h}h{eta_m:02d}m"
                else:
                    eta_str = "..."
                log(
                    f"Step {global_step}/{total_steps} | "
                    f"Loss: {ema_loss:.4f} | "
                    f"GradNorm: {grad_norm:.4f} | "
                    f"Tok/s: {tokens_per_sec:.0f} | "
                    f"ETA: {eta_str}"
                )
                step_start_time = time.time()

            if global_step % CHECKPOINT_INTERVAL == 0:
                optimizer.eval()
                save_stage_checkpoint(stage_module, optimizer, global_step, epoch,
                                      checkpoint_dir, rank, stage_info, tokenizer, config)
                optimizer.train()

        # Epoch-end validation
        stage_module.eval()
        optimizer.eval()
        val_loss = 0.0
        val_batches = 0
        for input_ids, labels in val_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            pipe_stage = PipelineStage(
                stage_module, stage_index=rank, num_stages=world_size, device=device,
            )
            schedule = ScheduleGPipe(pipe_stage, n_microbatches=N_MICROBATCHES, loss_fn=loss_fn)

            microbatch_losses = []
            if rank == 0:
                schedule.step(input_ids)
            elif rank == world_size - 1:
                schedule.step(target=labels, losses=microbatch_losses)
            else:
                schedule.step()

            loss_tensor = torch.zeros(1, device=device)
            if rank == world_size - 1 and microbatch_losses:
                loss_tensor[0] = sum(l.item() for l in microbatch_losses) / len(microbatch_losses)
            dist.broadcast(loss_tensor, src=world_size - 1)
            val_loss += loss_tensor.item()
            val_batches += 1

        if val_batches > 0:
            log(f"Epoch {epoch+1}/{NUM_EPOCHS}, Validation Loss: {val_loss / val_batches:.4f}")

        # Epoch-end checkpoint (with eval params)
        save_stage_checkpoint(stage_module, optimizer, global_step, epoch + 1,
                              checkpoint_dir, rank, stage_info, tokenizer, config)

    # Save final full model via from_pretrained-compatible format
    dist.barrier()
    if rank == 0:
        latest_ckpt = os.path.join(checkpoint_dir, f"checkpoint-{global_step}")
        full_model_path = os.path.join(latest_ckpt, "full_model.pt")
        if os.path.exists(full_model_path):
            full_state = torch.load(full_model_path, map_location="cpu", weights_only=False)
            model = RecursiveCompressorLM(config)
            model.load_state_dict(full_state)
            final_path = os.path.join(data_dir, "final_model")
            model.save_pretrained(final_path)
            tokenizer.save_pretrained(final_path)
            log(f"Final model saved: {final_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    train()
