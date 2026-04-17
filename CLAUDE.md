# CLAUDE.md

## Project Overview
Python ML project: RecursiveCompressor - a language model with recursive compression architecture.
Uses HuggingFace (PreTrainedModel), PyTorch DDP and pipeline parallelism, and uv for package management.

## Architecture
- `recursive_compressor.py` - Core module. Splits input into chunks, processes with causal attention, compresses inter-chunk information recursively. Has `step`/`forward`/`predict` methods.
- `recursive_compressor_lm.py` - Language model wrapping RecursiveCompressor layers. Extends HuggingFace PreTrainedModel.
- `recursive_compressor_lm_pipeline.py` - Pipeline stage wrapper that splits the LM across GPUs (first stage owns embedding, last stage owns norm+head, middle stages own layers).
- `configuration_recursive_compressor.py` - HuggingFace PretrainedConfig for model parameters.
- `dataset.py` - Data pipeline with memmap caching. Tokenizes HF datasets, packs short documents into context-length sequences.
- `train.py` - DDP training with RAdamScheduleFree, checkpointing, file-based control commands.
- `train_pipeline.py` - Pipeline parallel training using PyTorch PipelineStage + ScheduleGPipe. Checkpoints save per-stage `.pt` plus a reconstructed `full_model.pt` and `config.json`.
- `predict.py` - Text generation using `from_pretrained` and `step` method. Detects pipeline checkpoints (`full_model.pt`) and loads them via `config.json`.

## Commands
```bash
uv sync                                          # Install dependencies
uv run pytest test_lm.py -v                      # Run tests
uv run python train.py                           # Single GPU training (DDP-compatible)
uv run torchrun --nproc_per_node=6 train.py      # 6-GPU DDP training
uv run torchrun --nproc_per_node=6 train_pipeline.py  # 6-GPU pipeline parallel training
uv run python predict.py "text" --model-dir /path/to/checkpoint --context-length 256
```

## Training Control
```bash
echo "pause"         > control.cmd   # Pause all GPUs
echo "resume"        > control.cmd   # Resume
echo "save_and_exit" > control.cmd   # Save checkpoint and exit
```

## Environment
- `.env` file sets `DATA_DIR` (datasets, checkpoints, memmap caches)
- Default: `DATA_DIR=./data`
- Production: `DATA_DIR=/mnt/raid0/RecursiveCompressor`
- Hardware: 6x RTX 3090 (24GB VRAM each), 256GB RAM

## Key Design Decisions
- **Tokenizer**: `elyza/ELYZA-japanese-Llama-2-7b-fast`. `[DOC]`, `[QUERY]`, `[ANSWER]` are plain text markers, not special tokens.
- **Data format**: Documents: `<s>[DOC]text`, Conversations: `<s>[QUERY]q[ANSWER]a` per turn. BOS (`<s>`) separates units. No double-BOS.
- **Packing**: Short documents are concatenated into context-length sequences to reduce PAD waste. Unit = `[BOS] + tokens` (no trailing BOS). Trailing BOS added by packer.
- **Memmap caching**: Tokenized data stored as numpy memmap (uint16) for memory efficiency. Cache names include version suffix (e.g. `_v2`) - change suffix to force rebuild.
- **fp32 training**: Model trains in float32. bfloat16 was tried but caused instability with RAdamScheduleFree's `optimizer.eval()` extrapolation (`x = y/β - z(1/β-1)`). Padding `torch.zeros` should still inherit `dtype=x.dtype` for safety.
- **Numerical stability**: Use `F.scaled_dot_product_attention` (internally fp32 even for low-precision inputs, enables FlashAttention). Cast logits to float32 before CrossEntropyLoss.
- **schedule_free optimizer**: `optimizer.eval()` switches to averaged params "x" (better for inference). Must call `optimizer.train()` again before next `step()`. Periodic checkpoints sandwich the save with `eval()`/`train()`. Epoch-end save uses `eval()` and stays in eval (validation runs with eval params, then training proceeds in next epoch with `optimizer.train()`).
- **DDP**: Cache building happens before `init_process_group` (sentinel file sync). Control commands synced via `dist.broadcast`. Checkpoint barrier must be outside `if is_main_process()`.
- **Pipeline parallel**: All ranks see the same data (`DistributedSampler(num_replicas=1, rank=0)`). `ScheduleGPipe.step()` returns logits, not loss — collect microbatch losses via `losses=[]` argument. Per-stage checkpoint saves include rank-0-saved `config.json` and `full_model.pt` reconstructed via `reconstruct_full_state_dict`.
- **Sampler shuffle**: `DistributedSampler` defaults to `seed=0`; combined with `set_epoch(epoch)`, shuffle order is reproducible across restarts within the same epoch (so resume reads the same batches in the same order).

## Debugging Guidelines
- When modifying data pipeline (packing, collation), add shape/length assertions before and after transformations.
- All sequences in `_pack_units` must be exactly `context_length`. Use `(seq + [PAD] * context_length)[:context_length]` pattern to guarantee.
- After modifying training code, run `uv run pytest test_lm.py -v` before committing.

## Current Model Parameters
- d_model=2048, num_heads=16, d_ff=4096, chunk_size=4, compress_size=1, num_layers=12 (DDP) / 16 (pipeline)
- context_length=2048, optimizer=RAdamScheduleFree
- DDP: lr=3e-4, grad_accum=4. Pipeline: lr=1e-4, n_microbatches=6, batch_size=6
- dtype=float32

## Datasets (Japanese only)
- `wikimedia/wikipedia` (20231101.ja)
- `hotchpotch/cc100-ja-documents`
- `shi3z/ja_conv_wikipedia_llama2pro8b_30k`
- `shi3z/ja_conv_wikipedia_orion14B_100K`
