"""
Pipeline parallel training script for RecursiveCompressorLM.

Usage:
    uv run torchrun --nproc_per_node=6 train_pipeline.py

Control commands (write to control.cmd file during training):
    echo "pause"         > control.cmd   # Pause training
    echo "resume"        > control.cmd   # Resume training
    echo "save_and_exit" > control.cmd   # Save checkpoint and exit
"""

import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import Schedule1F1B
from dotenv import load_dotenv

from configuration_recursive_compressor import RecursiveCompressorConfig
from recursive_compressor_lm import RecursiveCompressorLM
from recursive_compressor_lm_pipeline import RecursiveCompressorLMPipelineStage
from dataset import DATASET_TYPES, prepare_all_datasets

torch.set_float32_matmul_precision("high")


# Names that should NOT use Muon (use AdamW instead).
# Muon is for 2D hidden-layer weights; embedding, output head, and learnable
# context/query vectors are kept on AdamW per the original Muon recipe.
_ADAMW_ONLY_KEYWORDS = ("embedding", "head", "compressor_query", "initial_context")


def split_params_for_muon(model):
    """Split parameters into (muon_params, adamw_params).
    Muon manages 2D Linear weights inside hidden layers.
    AdamW manages embedding, output head, learnable contexts, biases, LayerNorms."""
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

load_dotenv()

torch.set_float32_matmul_precision("high")

# Hyperparameters
CONTEXT_LENGTH = 2048
LEARNING_RATE = 5e-5
NUM_EPOCHS = 1
GRAD_CLIP = 1.0
N_MICROBATCHES = 6
BATCH_SIZE = 6  # Must be >= N_MICROBATCHES
CHECKPOINT_INTERVAL = 1000
MAX_CHECKPOINTS = 2
CONTROL_FILE = "control.cmd"
LOG_INTERVAL = 10
DATASET_PREFAULT = True  # Prefault memmap pages into OS page cache (shared across ranks)
CACHE_BUILD_WORKERS = 6  # Parallel tokenization workers when building memmap caches


def get_data_dir():
    return os.environ.get("DATA_DIR", "./data")


def get_checkpoint_dir():
    return os.path.join(get_data_dir(), "checkpoints_pipeline")


def _checkpoint_step(name):
    """`checkpoint-pretrain-13000` -> 13000"""
    return int(name.rsplit("-", 1)[1])


def _checkpoint_name(dataset_type, step):
    return f"checkpoint-{dataset_type}-{step}"


def _list_checkpoints(checkpoint_dir, dataset_type):
    """Return sorted-by-step list of checkpoint dir names matching dataset_type."""
    if not os.path.exists(checkpoint_dir):
        return []
    prefix = f"checkpoint-{dataset_type}-"
    names = [d for d in os.listdir(checkpoint_dir) if d.startswith(prefix)]
    return sorted(names, key=_checkpoint_step)


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


def save_stage_checkpoint(stage_module, optimizers, step, epoch, checkpoint_dir,
                          rank, stage_info, dataset_type,
                          tokenizer=None, config=None):
    """Save per-stage checkpoint. optimizers is a list of optimizer instances."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, _checkpoint_name(dataset_type, step))
    os.makedirs(path, exist_ok=True)

    torch.save({
        "stage_state_dict": stage_module.state_dict(),
        "optimizers_state_dict": [opt.state_dict() for opt in optimizers],
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
        save_full_model(path)
        _rotate_checkpoints(checkpoint_dir, dataset_type)


def save_full_model(checkpoint_path):
    """Gather all stage state dicts and save as full model."""
    world_size = dist.get_world_size()
    gathered = []
    for r in range(world_size):
        stage_path = os.path.join(checkpoint_path, f"stage_{r}.pt")
        data = torch.load(stage_path, map_location="cpu", weights_only=False)
        gathered.append((data["rank"], data["stage_info"], data["stage_state_dict"]))

    full_state = RecursiveCompressorLMPipelineStage.reconstruct_full_state_dict(gathered)
    torch.save(full_state, os.path.join(checkpoint_path, "full_model.pt"))


def _rotate_checkpoints(checkpoint_dir, dataset_type):
    """Keep only the latest MAX_CHECKPOINTS for this dataset_type."""
    checkpoints = _list_checkpoints(checkpoint_dir, dataset_type)
    while len(checkpoints) > MAX_CHECKPOINTS:
        old = os.path.join(checkpoint_dir, checkpoints.pop(0))
        shutil.rmtree(old)
        log(f"Removed old checkpoint: {old}")


def load_latest_checkpoint(stage_module, optimizers, checkpoint_dir, rank, dataset_type):
    """Load latest per-stage checkpoint matching dataset_type.
    Returns (step, epoch). If none found, returns (0, 0)."""
    checkpoints = _list_checkpoints(checkpoint_dir, dataset_type)
    if not checkpoints:
        return 0, 0

    latest = os.path.join(checkpoint_dir, checkpoints[-1])
    stage_path = os.path.join(latest, f"stage_{rank}.pt")

    if os.path.exists(stage_path):
        log(f"Resuming from checkpoint: {latest}")
        data = torch.load(stage_path, map_location="cpu", weights_only=False)
        stage_module.load_state_dict(data["stage_state_dict"])
        if "optimizers_state_dict" in data:
            for opt, state in zip(optimizers, data["optimizers_state_dict"]):
                opt.load_state_dict(state)
        else:
            log("Old optimizer format detected; skipping optimizer state (starting fresh).")
        return data["step"], data["epoch"]

    # Fall back to full model in latest checkpoint
    full_path = os.path.join(latest, "full_model.pt")
    if os.path.exists(full_path):
        log(f"Resuming from full model: {full_path}")
        full_state = torch.load(full_path, map_location="cpu", weights_only=False)
        stage_module.load_from_full_model(full_state)
        return 0, 0

    return 0, 0


def load_start_checkpoint(stage_module, start_checkpoint_path):
    """Load model weights from an arbitrary checkpoint dir (model only, no optimizer).
    Used for warm-starting fine-tuning from a pretrain checkpoint."""
    full_path = os.path.join(start_checkpoint_path, "full_model.pt")
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"full_model.pt not found in {start_checkpoint_path}")
    log(f"Warm-starting from: {full_path}")
    full_state = torch.load(full_path, map_location="cpu", weights_only=False)
    stage_module.load_from_full_model(full_state)


def train(dataset_type="pretrain", start_checkpoint=None):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    data_dir = get_data_dir()
    checkpoint_dir = get_checkpoint_dir()
    cache_dir = os.path.join(data_dir, "hf_cache")

    log(f"Dataset type: {dataset_type}")

    # Dataset cache (rank 0 builds, others wait)
    sentinel_path = os.path.join(cache_dir, "mmap", f".cache_ready_{dataset_type}")
    if rank == 0:
        print("Preparing datasets...", flush=True)
        full_dataset, tokenizer = prepare_all_datasets(
            CONTEXT_LENGTH, cache_dir=cache_dir,
            prefault=DATASET_PREFAULT, dataset_type=dataset_type,
            num_workers=CACHE_BUILD_WORKERS,
        )
        with open(sentinel_path, "w") as f:
            f.write("ready")
        print("Cache ready.", flush=True)
    else:
        while not os.path.exists(sentinel_path):
            time.sleep(2)
        # Other ranks just open the memmap (page cache populated by rank 0 if prefault enabled)
        full_dataset, tokenizer = prepare_all_datasets(
            CONTEXT_LENGTH, cache_dir=cache_dir,
            prefault=False, dataset_type=dataset_type,
            num_workers=1,
        )

    dist.barrier()
    if rank == 0 and os.path.exists(sentinel_path):
        os.remove(sentinel_path)

    config = RecursiveCompressorConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=2048,
        num_heads=16,
        d_ff=6144,
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
    # Master weights and optimizer state stay in fp32; forward/backward use
    # bfloat16 via autocast (mixed precision).
    stage_module = RecursiveCompressorLMPipelineStage(config, **stage_info).to(device)

    stage_params = sum(p.numel() for p in stage_module.parameters())
    total_params_tensor = torch.tensor([stage_params], dtype=torch.long, device=device)
    dist.all_reduce(total_params_tensor)
    log(f"Total layers: {config.num_layers}, Stages: {world_size}, Total params: {total_params_tensor.item():,}, "
        f"params dtype: fp32, compute dtype: bfloat16 (autocast)")
    log(f"Stage {rank}: layers {stage_info['layer_start']}-{stage_info['layer_end']}, "
        f"params: {stage_params:,}, first={stage_info['is_first']}, last={stage_info['is_last']}")

    # Data (all ranks see the same data — pipeline parallel, not data parallel)
    train_dataset = full_dataset
    log(f"Train: {len(train_dataset)}")

    train_sampler = DistributedSampler(train_dataset, num_replicas=1, rank=0, shuffle=True)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        sampler=train_sampler, num_workers=2, pin_memory=True, drop_last=True,
    )

    # Optimizers: Muon for hidden 2D matrices, AdamW for embedding/head/biases/norms
    muon_params, adamw_params = split_params_for_muon(stage_module)
    optimizers = []
    if muon_params:
        optimizers.append(torch.optim.Muon(
            muon_params, lr=LEARNING_RATE, weight_decay=0.0,
            momentum=0.95, adjust_lr_fn="match_rms_adamw",
        ))
    if adamw_params:
        optimizers.append(torch.optim.AdamW(
            adamw_params, lr=LEARNING_RATE, weight_decay=0.0,
        ))
    log(f"Stage {rank}: Muon manages {sum(p.numel() for p in muon_params):,} params, "
        f"AdamW manages {sum(p.numel() for p in adamw_params):,} params")

    # Resume from existing checkpoint of the same dataset_type if any
    existing = _list_checkpoints(checkpoint_dir, dataset_type)
    if existing and start_checkpoint is not None:
        log(f"WARNING: --start-checkpoint={start_checkpoint} ignored because "
            f"{len(existing)} existing {dataset_type} checkpoint(s) found in {checkpoint_dir}. "
            f"Resuming from latest.")
    start_step, start_epoch = load_latest_checkpoint(
        stage_module, optimizers, checkpoint_dir, rank, dataset_type,
    )
    if not existing and start_checkpoint is not None:
        # Warm-start model weights from external checkpoint (no optimizer state)
        load_start_checkpoint(stage_module, start_checkpoint.rstrip("/"))
    global_step = start_step

    # Loss function (only used on last stage)
    vocab_size = config.vocab_size

    def loss_fn(logits, targets):
        return nn.CrossEntropyLoss()(logits.float().view(-1, vocab_size), targets.view(-1))

    # Training loop
    total_steps = len(train_loader)
    for epoch in range(start_epoch, NUM_EPOCHS):
        stage_module.train()
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
                save_stage_checkpoint(stage_module, optimizers, global_step, epoch,
                                      checkpoint_dir, rank, stage_info, dataset_type,
                                      tokenizer=tokenizer, config=config)
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
                    save_stage_checkpoint(stage_module, optimizers, global_step, epoch,
                                          checkpoint_dir, rank, stage_info, tokenizer, config)
                    dist.destroy_process_group()
                    return

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # Build pipeline schedule for this step
            pipe_stage = PipelineStage(
                stage_module, stage_index=rank, num_stages=world_size, device=device,
            )
            schedule = Schedule1F1B(pipe_stage, n_microbatches=N_MICROBATCHES, loss_fn=loss_fn)

            # Execute pipeline (losses are collected via the losses list arg).
            # Forward/backward run in bfloat16 via autocast; LayerNorm, Softmax,
            # and CE loss stay in fp32 by autocast's policy. Gradients flow back
            # to fp32 master weights.
            microbatch_losses = []
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if rank == 0:
                    schedule.step(input_ids)
                elif rank == world_size - 1:
                    schedule.step(target=labels, losses=microbatch_losses)
                else:
                    schedule.step()

            # Gradient clipping and optimizer step
            grad_norm = nn.utils.clip_grad_norm_(stage_module.parameters(), GRAD_CLIP).item()
            for opt in optimizers:
                opt.step()
            for opt in optimizers:
                opt.zero_grad()

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
                save_stage_checkpoint(stage_module, optimizers, global_step, epoch,
                                      checkpoint_dir, rank, stage_info, dataset_type,
                                      tokenizer=tokenizer, config=config)

        # Epoch-end checkpoint
        save_stage_checkpoint(stage_module, optimizers, global_step, epoch + 1,
                              checkpoint_dir, rank, stage_info, dataset_type,
                              tokenizer=tokenizer, config=config)

    # Save final full model via from_pretrained-compatible format
    dist.barrier()
    if rank == 0:
        latest_ckpt = os.path.join(checkpoint_dir, _checkpoint_name(dataset_type, global_step))
        full_model_path = os.path.join(latest_ckpt, "full_model.pt")
        if os.path.exists(full_model_path):
            full_state = torch.load(full_model_path, map_location="cpu", weights_only=False)
            model = RecursiveCompressorLM(config)
            model.load_state_dict(full_state)
            final_path = os.path.join(data_dir, f"final_model_{dataset_type}")
            model.save_pretrained(final_path)
            tokenizer.save_pretrained(final_path)
            log(f"Final model saved: {final_path}")

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Pipeline parallel training for RecursiveCompressorLM")
    parser.add_argument("--dataset-type", choices=list(DATASET_TYPES), default="pretrain",
                        help="使用するデータセット種別")
    parser.add_argument("--start-checkpoint", type=str, default=None,
                        help="ファインチューニング元のチェックポイントディレクトリ "
                             "（同 dataset_type のチェックポイントが既存の場合は無視され警告）")
    args = parser.parse_args()
    train(dataset_type=args.dataset_type, start_checkpoint=args.start_checkpoint)


if __name__ == "__main__":
    main()
