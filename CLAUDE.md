# CLAUDE.md

## Project Overview
Python ML project: RecursiveCompressor - a language model with recursive compression architecture.
Uses HuggingFace (PreTrainedModel), PyTorch pipeline parallelism, and uv for package management.

## Architecture
- `recursive_compressor.py` - Core module. Splits input into chunks, processes with causal attention (now GatedAttention), compresses inter-chunk information recursively. Has `step`/`forward`/`predict` methods.
- `recursive_compressor_lm.py` - Language model wrapping RecursiveCompressor layers. Extends HuggingFace PreTrainedModel.
- `recursive_compressor_lm_pipeline.py` - Pipeline stage wrapper that splits the LM across GPUs (first stage owns embedding, last stage owns norm+head, middle stages own layers).
- `configuration_recursive_compressor.py` - HuggingFace PretrainedConfig for model parameters.
- `dataset.py` - Data pipeline with memmap caching. Tokenizes HF datasets, packs short documents into context-length sequences.
- `train_pipeline.py` - Pipeline parallel training using PyTorch PipelineStage + Schedule1F1B with Muon + AdamW. Mixed precision (fp32 master weights, bfloat16 autocast). Checkpoints save per-stage `.pt` plus a reconstructed `full_model.pt` and `config.json`.
- `predict.py` - Text generation using `from_pretrained` and `step` method. Detects pipeline checkpoints (`full_model.pt`) and loads them via `config.json`. Supports temperature and top-p sampling.
- `predict_stream.py` - Interactive REPL with token-by-token streaming via HuggingFace `TextStreamer`. Same sampling options.

## Commands
```bash
uv sync                                                # Install dependencies
uv run pytest test_lm.py -v                            # Run tests
uv run torchrun --nproc_per_node=6 train_pipeline.py   # 6-GPU pipeline parallel training
uv run python predict.py "text" --model-dir /path/to/checkpoint --context-length 256 --top-p 0.9
uv run python predict_stream.py --model-dir /path/to/checkpoint --top-p 0.9
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
- **Tokenizer**: `elyza/ELYZA-japanese-Llama-2-7b-fast`. `[QUERY]`, `[ANSWER]` are plain text markers (no `[DOC]` prefix anymore — documents are just `<s>text`).
- **Data format**: Documents: `<s>text` (raw, no marker). Conversations: `<s>[QUERY]q[ANSWER]a` per turn.
- **Chunking + Packing**: Each text is BOS-prefixed and split into `context_length`-sized chunks. The first chunk has `<s>`; continuation chunks are unmarked. Chunks are concatenated to fill context_length samples; no trailing BOS is added (the next chunk's leading BOS serves as separator). Pad with PAD tokens.
- **Memmap caching**: Tokenized data stored as numpy memmap (uint16) for memory efficiency. Cache names include version suffix (e.g. `_v3`) - change suffix to force rebuild. `prefault=True` reads through memmap once on rank 0 to populate OS page cache (shared across all ranks; not per-process copy).
- **Mixed precision**: Master weights and optimizer state in fp32, forward/backward in bfloat16 via `torch.autocast`. LayerNorm and Softmax stay in fp32 by autocast policy. CrossEntropyLoss receives `logits.float()` cast.
- **Numerical stability**: Use `F.scaled_dot_product_attention` (internally fp32 even for low-precision inputs, enables FlashAttention). Padding `torch.zeros` must inherit `dtype=x.dtype`.
- **Optimizers**: Muon (`torch.optim.Muon`) for 2D hidden Linear weights with `adjust_lr_fn="match_rms_adamw"`; AdamW for embedding, head, learnable contexts (`compressor_query`, `initial_context`), biases, and LayerNorms. Both share the same LR. `split_params_for_muon()` does the partition.
- **Pipeline parallel**: All ranks see the same data (`DistributedSampler(num_replicas=1, rank=0)`). `Schedule1F1B.step()` returns logits, not loss — collect microbatch losses via `losses=[]` argument. Per-stage checkpoint saves include rank-0-saved `config.json` and `full_model.pt` reconstructed via `reconstruct_full_state_dict`. Old `optimizer_state_dict` checkpoints are detected and skipped (model weights load, optimizers start fresh).
- **Sampler shuffle**: `DistributedSampler` defaults to `seed=0`; combined with `set_epoch(epoch)`, shuffle order is reproducible across restarts within the same epoch (so resume reads the same batches in the same order).
- **Cache build sync**: Cache building happens before `init_process_group` (sentinel file). Control commands synced via `dist.broadcast`. All ranks must call `dist.barrier()` after checkpoint save (not inside `if rank == 0`).

## Debugging Guidelines
- When modifying data pipeline (packing, collation), add shape/length assertions before and after transformations.
- All sequences in `_pack_chunks` must be exactly `context_length`. Use `(seq + [PAD] * context_length)[:context_length]` pattern to guarantee.
- After modifying training code, run `uv run pytest test_lm.py -v` before committing.
- `test_step_split_consistency` and `test_predict_matches_forward` use `atol=1e-3, rtol=1e-2` — tightened tolerances may fail on numerical drift from architecture changes (GatedAttention etc.).

## Current Model Parameters
- d_model=2048, num_heads=16, d_ff=6144, chunk_size=4, compress_size=1, num_layers=16
- context_length=2048, lr=5e-5
- optimizers: Muon (2D hidden weights) + AdamW (embedding/head/biases/norms)
- pipeline: n_microbatches=6, batch_size=6
- mixed precision: fp32 master / bfloat16 autocast

## Datasets
Selected by `--dataset-type` in `train_pipeline.py`:
- `pretrain` (documents only):
  - `wikimedia/wikipedia` (20231101.ja, 20231101.en)
  - `hotchpotch/cc100-ja-documents`
  - `JeanKaddour/minipile`
- `instruct` (conversations only):
  - `shi3z/ja_conv_wikipedia_llama2pro8b_30k`
  - `shi3z/ja_conv_wikipedia_orion14B_100K`
  - `HuggingFaceH4/ultrachat_200k`
