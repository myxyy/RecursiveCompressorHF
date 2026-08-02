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

## TensorBoard
Raw per-step `train/loss` and `train/grad_norm` are logged to
`$DATA_DIR/tensorboard/{dataset_type}/`. View with:
```bash
uv run tensorboard --logdir $DATA_DIR/tensorboard/
```

## Environment
- `.env` file sets `DATA_DIR` (datasets, checkpoints, memmap caches)
- Default: `DATA_DIR=./data`
- Production: `DATA_DIR=/mnt/raid0/RecursiveCompressor`
- Hardware: 6x RTX 3090 (24GB VRAM each), 256GB RAM

## Key Design Decisions
- **Tokenizer**: `elyza/ELYZA-japanese-Llama-2-7b-fast`. `[INST]`, `[/INST]` are Llama-2-style plain text markers (not special tokens).
- **Data format** (Llama 2 style): Documents: `<s>text</s>`. Conversations: `<s>[INST]q1[/INST]a1</s><s>[INST]q2[/INST]a2</s>...` (each turn is BOS+EOS-wrapped).
- **Pretrain chunking + packing** (`_build_memmap_packed`): Each text is wrapped as `[BOS] + tokens + [EOS]` and split into `context_length`-sized chunks. The first chunk starts with `<s>`, the last chunk ends with `</s>`; if the text fits in one chunk it has both. Continuation chunks are unmarked. Chunks are concatenated (short-text packing) to fill context_length samples; pad with PAD tokens. Loss on all non-PAD positions.
- **Instruct conversations** (`_build_memmap_conversations`): Each conversation is tokenized turn-by-turn — prompt `<s>[INST]q[/INST]` and answer `a</s>` are tokenized *separately* (so the loss-mask boundary is exact and matches inference, where chat_server feeds `<s>[INST]q[/INST]` and generates the answer). One conversation → its own context_length sample(s); **no cross-conversation packing** (a long conversation is split into chunks, a short one is padded). An `answer-only loss mask` (uint8, 1 = answer token incl. its EOS, 0 = prompt/PAD) is stored in a parallel `.mask` memmap; `meta.has_mask=True`. `MemmapDataset.__getitem__` masks non-answer label positions to -100 when a mask is present (pretrain caches have no mask → loss on all non-PAD, unchanged). Samples with no answer token in a label position are dropped (NaN guard).
- **Memmap caching**: Tokenized data stored as numpy memmap (uint16) for memory efficiency. Caches live under `$DATA_DIR/hf_cache/mmap/ctx{context_length}/` — the per-context-length subfolder means changing `CONTEXT_LENGTH` builds a fresh set without clobbering caches for other lengths. Cache version suffix is **per-source** (pretrain still `_v5`; instruct bumped to `_v6` because the answer-mask added a format change — pretrain caches were left at `_v5` to avoid a multi-million-sample rebuild). Change a source's suffix to force its rebuild. `prefault=True` reads through memmap once on rank 0 to populate OS page cache (shared across all ranks; not per-process copy).
- **All-PAD-label NaN guard**: `_pack_chunks` enforces `MIN_CONTENT=2` so a packed sample always has ≥1 valid label position (otherwise CE loss = 0/0 = NaN). Loss functions in `train_pipeline.loss_fn` and `RecursiveCompressorLM.forward` ALSO guard against all-PAD microbatches (return `logits.sum() * 0.0`) for defense-in-depth. Discovered when training hit NaN deterministically at ~step 12610 from a single 1-token continuation chunk that became its own sample.
- **Mixed precision**: Master weights and optimizer state in fp32, forward/backward in bfloat16 via `torch.autocast`. Softmax stays in fp32 by autocast policy. RMSNorm computes internally in fp32 (weight is fp32) but **outputs bf16** — unlike LayerNorm which is in autocast's fp32-output list. CrossEntropyLoss receives `logits.float()` cast.
- **Numerical stability**: Use `F.scaled_dot_product_attention` (internally fp32 even for low-precision inputs, enables FlashAttention). Padding `torch.zeros` must inherit `dtype=x.dtype`.
- **Optimizers**: Muon (`torch.optim.Muon`) for 2D hidden Linear weights with `adjust_lr_fn="match_rms_adamw"`; AdamW for embedding, head, learnable contexts (`compressor_query`, `initial_context`), biases, and RMSNorms. Both share the same LR. `split_params_for_muon()` does the partition by `param.ndim >= 2`, so 1D RMSNorm weights naturally fall on AdamW. (An AdamCScheduleFreePlusPaper experiment underperformed Muon and used more VRAM; reverted, code remains on the `warmup-schedulefree-plus` branch.)
- **LR scheduler**: `ReduceLROnPlateau` applied **per step** (not per epoch), fed the EMA loss (`EMA_BETA=0.99` — the EMA stands in for an epoch average to smooth per-step noise). One scheduler per optimizer, both stepped with the same EMA loss each step so they stay in lockstep across optimizers and ranks (loss is broadcast before use). `factor=0.9, patience=1000, cooldown=100`. Scheduler state (best/bad-step/cooldown counters) is saved in per-stage checkpoints as `schedulers_state_dict` and restored on resume (the reduced lr itself lives in optimizer state); old checkpoints without it just reset the counters. Current lr is logged to TensorBoard as `train/lr` and lr reductions are printed to the console.
- **Pipeline parallel**: All ranks see the same data (`DistributedSampler(num_replicas=1, rank=0)`). `Schedule1F1B.step()` returns logits, not loss — collect microbatch losses via `losses=[]` argument. Per-stage checkpoint saves include rank-0-saved `config.json` and `full_model.pt` reconstructed via `reconstruct_full_state_dict`. Old `optimizer_state_dict` checkpoints are detected and skipped (model weights load, optimizers start fresh).
- **VRAM balance / STAGE_LAYER_SPLIT**: Under 1F1B, stage r holds `num_stages - r` in-flight microbatch activations (rank 0: 6, rank 5: 1) — the dominant source of per-GPU VRAM imbalance (~0.6 GB per layer·microbatch at ctx1024/d2048, vs ~1.2 GB fp32 params+grad+momentum per layer; embedding/head add ~1.5 GB to the edge stages). `STAGE_LAYER_SPLIT` in train_pipeline.py skews layer counts toward later stages to compensate (must sum to num_layers; None = even split). Tune against observed per-GPU VRAM. Changing the split mid-run is safe: resume detects the layout change via the checkpoint's `stage_info` and reloads weights from `full_model.pt` (optimizer state starts fresh; step/EMA/scheduler counters carry over).
- **Sampler shuffle**: `DistributedSampler` defaults to `seed=0`; combined with `set_epoch(epoch)`, shuffle order is reproducible across restarts within the same epoch (so resume reads the same batches in the same order).
- **Cache build sync**: Cache building happens before `init_process_group` (sentinel file). Control commands synced via `dist.broadcast`. All ranks must call `dist.barrier()` after checkpoint save (not inside `if rank == 0`).

## Debugging Guidelines
- When modifying data pipeline (packing, collation), add shape/length assertions before and after transformations.
- All sequences in `_pack_chunks` must be exactly `context_length`. Use `(seq + [PAD] * context_length)[:context_length]` pattern to guarantee.
- After modifying training code, run `uv run pytest test_lm.py -v` before committing.
- `test_step_split_consistency` and `test_predict_matches_forward` use `atol=5e-3, rtol=5e-2` — these check numerical equivalence which is amplified by data-dependent xs propagation (`comp_query_out` carrying through layers). Architecture changes that strengthen this dependency (e.g. cross-attention residuals on the query side) require loosening tolerances further.
- **NaN debugging**: training is deterministic with `seed=0` + `set_epoch(epoch)`, so resuming from a checkpoint reproduces the same NaN at the same step. To inspect which sample/layer first produces NaN, write a single-GPU debug script that loads the checkpoint, runs forward on the failing batch with per-sample iteration, and uses forward hooks to detect NaN/Inf at each layer output. **Loss=NaN with finite GradNorm typically means logits are clean but loss has 0/0** (all-PAD labels in a microbatch).

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
