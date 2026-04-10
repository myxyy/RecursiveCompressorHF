# CLAUDE.md

## Project Overview
Python ML project: RecursiveCompressor - a language model with recursive compression architecture.
Uses HuggingFace (PreTrainedModel), PyTorch DDP, and uv for package management.

## Architecture
- `recursive_compressor.py` - Core module. Splits input into chunks, processes with causal attention, compresses inter-chunk information recursively. Has `step`/`forward`/`predict` methods.
- `recursive_compressor_lm.py` - Language model wrapping RecursiveCompressor layers. Extends HuggingFace PreTrainedModel.
- `configuration_recursive_compressor.py` - HuggingFace PretrainedConfig for model parameters.
- `dataset.py` - Data pipeline with memmap caching. Tokenizes HF datasets, packs short documents into context-length sequences.
- `train.py` - DDP training with RAdamScheduleFree, checkpointing, file-based control commands.
- `predict.py` - Text generation using `from_pretrained` and `step` method.

## Commands
```bash
uv sync                                    # Install dependencies
uv run pytest test_lm.py -v               # Run tests
uv run python train.py                     # Single GPU training
uv run torchrun --nproc_per_node=6 train.py  # 6-GPU DDP training
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
- **bfloat16**: Model runs in bfloat16. `torch.zeros` for padding must specify `dtype=x.dtype`.
- **schedule_free optimizer**: Do NOT call `optimizer.eval()` before saving checkpoints - it corrupts parameters in bfloat16 due to extrapolation. `model.eval()` alone suffices for validation.
- **DDP**: Cache building happens before `init_process_group` (sentinel file sync). Control commands synced via `dist.broadcast`. Checkpoint barrier must be outside `if is_main_process()`.

## Debugging Guidelines
- When modifying data pipeline (packing, collation), add shape/length assertions before and after transformations.
- All sequences in `_pack_units` must be exactly `context_length`. Use `(seq + [PAD] * context_length)[:context_length]` pattern to guarantee.
- After modifying training code, run `uv run pytest test_lm.py -v` before committing.

## Current Model Parameters
- d_model=2048, num_heads=16, d_ff=4096, chunk_size=8, compress_size=4, num_layers=8
- context_length=2048, optimizer=RAdamScheduleFree, lr=3e-4, grad_accum=4
- dtype=bfloat16

## Datasets (Japanese only)
- `wikimedia/wikipedia` (20231101.ja)
- `hotchpotch/cc100-ja-documents`
- `shi3z/ja_conv_wikipedia_llama2pro8b_30k`
- `shi3z/ja_conv_wikipedia_orion14B_100K`
