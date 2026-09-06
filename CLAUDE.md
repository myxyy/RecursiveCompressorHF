# CLAUDE.md

## Project Overview
Python ML project: a language model with a custom hierarchical-kv-compression architecture (**LogKV**, the current main line). The previous recursive-compression architecture (RecursiveCompressor) is retained as legacy. Uses HuggingFace (PreTrainedModel), PyTorch DDP, and uv for package management.

## Architecture
### LogKV (main)
- `logkv.py` - Core module. Per level i (sub-unit = C^i tokens), each query attends only to completed sub-units in its current block (c < j); these disjoint intervals partition the entire past. All levels share one softmax (at most C−1 slots per level). See `logkv-refine.drawio.png` and `doc/logkv.md` §6.17. Compression is attention pooling with the chunk-last query. Has `forward`/`step`/`predict` (fp64 machine-precision equivalent) plus `LogKVBlock` (pre-norm attention+FFNSwiGLU) and options: `phase_emb`/`phase_levels`, `gated_attention`, `self_slot`, `learnable_decay`, `kv_norm`, `v_norm_only`, `level_amplify`.
- `logkv_lm.py` - LogKVLM language model (PreTrainedModel + generate; `past_key_values` carries the opaque per-layer hidden list).
- `configuration_logkv.py` - LogKVConfig.
- `train_logkv.py` - **DDP data-parallel** training (the model fits on one GPU). Muon + AdamW, bf16 autocast, control.cmd, `--resume latest` (skips consumed data, absolute `--max-steps`, EMA carry-over), periodic Japanese sample generations to `samples.log`.
- `predict_logkv.py` - Text generation for LogKV checkpoints.
- `exp/copying/`, `exp/selective-copying/` - Copy Memory Problem / Selective Copying suites (`--arch logkv` supported; selective wraps copying via task-module injection).
- `doc/logkv.md` - **The design/experiment record for LogKV. Read this first for any LogKV work.**

### Shared
- `dataset.py` - Data pipeline with memmap caching. Tokenizes HF datasets, packs short documents into context-length sequences.
- `predict.py` / `predict_stream.py` - Generation / interactive streaming REPL. `_load_model` picks the architecture from config.json's `model_type` ("logkv" → LogKVLM); also detects legacy pipeline checkpoints (`full_model.pt`).
- `chat_server.py` - Chat web UI (legacy-model era; decaying repetition penalty, reset/interrupt).

### Legacy (RecursiveCompressor)
- `recursive_compressor.py`, `recursive_compressor_lm.py`, `recursive_compressor_lm_pipeline.py`, `configuration_recursive_compressor.py`, `train_pipeline.py` (6-GPU pipeline parallel, Schedule1F1B). History: `doc/copying-memory-branch-changes.md`; full experiment logs under `doc/instruction-for-claude/`.

## Commands
```bash
uv sync                                                # Install dependencies
uv run pytest test_logkv.py test_logkv_lm.py -v        # LogKV tests
uv run pytest test_lm.py -v                            # Legacy tests

# LogKV standard-config training (DDP, 6 GPUs)
uv run torchrun --nproc_per_node=6 train_logkv.py --run-name <name> \
    --phase-emb --phase-levels 2 --gated-attention --self-slot

uv run python predict_logkv.py --model-dir $DATA_DIR/checkpoints_logkv/<name>/checkpoint-<step>/model \
    --max-new-tokens 1024 --temperature 0.7 --top-p 0.9
uv run python predict_stream.py --model-dir /path/to/checkpoint --temperature 0.7 --top-p 0.9

uv run torchrun --nproc_per_node=6 train_pipeline.py   # legacy pipeline-parallel training
```

## Training Control
```bash
just pause / just resume / just save-and-exit   # writes control.cmd (pause keeps GPUs allocated but idle)
```
Resume flags must match the run's original flags (model structure comes from the checkpoint's config.json).

## TensorBoard
`train/loss`, `train/grad_norm`, `train/lr` per step: LogKV runs under `$DATA_DIR/tensorboard/logkv-{dataset_type}/{run}/`, legacy under `$DATA_DIR/tensorboard/{dataset_type}/`.
```bash
uv run tensorboard --logdir $DATA_DIR/tensorboard/
```

## Environment
- `.env` file sets `DATA_DIR` (datasets, checkpoints, memmap caches)
- Default: `DATA_DIR=./data`; Production: `DATA_DIR=/mnt/raid0/RecursiveCompressor`
- Hardware: 6x RTX 3090 (24GB VRAM each), 256GB RAM

## Key Design Decisions — LogKV
Details and evidence live in `doc/logkv.md`; summary:
- **Refined layout (2026-09-06)** removes overlapping previous-block slots. The quality/throughput measurements below are historical, from the overlapping layout; refined training and generation quality need fresh evaluation. Weight shapes/config flags are unchanged, so old checkpoints load with new attention semantics. Restart generation with hidden=None; old runtime hidden states are incompatible.
- **Standard config**: fixed level decay (−i·log C) + phase embedding (levels=2, period 16) + multi-head + gated attention + self slot (the query's own k/v as one extra slot = standard causal-mask semantics; loss-neutral, gives the softmax an "attend to nothing" option, stabilizes grad_norm ~1.14→0.74). `kv_norm`/`learnable_decay`/`level_amplify`/`v_norm_only` exist as options but are NOT standard (each was tested and rejected for the LM: kv_norm caps the key-norm retrieval margin, learnable decay and amplification worsen temp-0.7 repetition, v_norm alone helps little).
- **Level decay** originally corrected cross-level multiplicity and improved topic fixation. It is retained as a coarse-level penalty in the refined layout, where slots no longer overlap; its former multiplicity rationale no longer applies. The 16.7M-token Copying result (8273× the training horizon) belongs to the original layout.
- **Phase embedding** (base-C digits of the absolute position, small period) breaks the positional degeneracy inside runs of identical tokens (multi-scale windows coincide there, making "how many so far" uncountable). Longer periods alias beyond the training range and hurt extrapolation; period 16 is the robust choice.
- **step()/forward()/predict() equivalence** is the core invariant, tested at fp64 <1e-12 against an independent `reference_forward` in `test_logkv.py` (forward delegates to step, so the test oracle must stay independent). Any semantic change (biases, norms, gates) must be mirrored in the reference.
- **Hidden format**: `(levels, offset)`; `levels[i] = [cur_q, cur_k, cur_v]` holds only the unfinished chunk (<C entries), with heads folded into the batch dim. Offset reconstructs absolute positions (phase, block bases). Retained slices are cloned to avoid pinning full-segment storage during inference. O(C·log L·d) memory.
- **Online softmax + activation checkpointing**: the whole attention pass is one non-reentrant checkpoint region; autograd keeps only per-level contexts (~1.33·L·d) instead of (L, C·levels, d) slot gathers. bf16-weight inference (no autocast) needs the softmax weights cast to the value dtype (fixed; see test).
- VRAM/throughput at d1024/8H/16L/ctx2048 (~310M params): batch 4/GPU ≈ 16 GiB, ~16-17k tok/s total on 6 GPUs (~4h per 5000 steps).

## Key Design Decisions — data pipeline & training (shared)
- **Tokenizer**: `elyza/ELYZA-japanese-Llama-2-7b-fast`. `[INST]`, `[/INST]` are Llama-2-style plain text markers (not special tokens).
- **Data format** (Llama 2 style): Documents: `<s>text</s>`. Conversations: `<s>[INST]q1[/INST]a1</s><s>[INST]q2[/INST]a2</s>...` (each turn BOS+EOS-wrapped).
- **Pretrain chunking + packing** (`_build_memmap_packed`): `[BOS] + tokens + [EOS]` split into context_length chunks (first chunk has `<s>`, last has `</s>`, continuations unmarked); chunks packed to fill samples, PAD-filled; loss on all non-PAD positions.
- **Instruct conversations** (`_build_memmap_conversations`): prompt `<s>[INST]q[/INST]` and answer `a</s>` tokenized separately for an exact loss-mask boundary; answer-only loss mask in a parallel `.mask` memmap; no cross-conversation packing.
- **Memmap caching**: uint16 memmaps under `$DATA_DIR/hf_cache/mmap/ctx{context_length}/`; per-source version suffix (pretrain `_v5`, instruct `_v6`) — bump a suffix to force rebuild. `prefault=True` warms the OS page cache on rank 0 (shared across ranks).
- **All-PAD-label NaN guard**: `_pack_chunks` enforces `MIN_CONTENT=2`; loss functions also return `logits.sum() * 0.0` for all-PAD (micro)batches (0/0 CE = NaN otherwise).
- **Mixed precision**: fp32 master weights/optimizer state, bf16 autocast forward/backward. RMSNorm computes in fp32 but outputs bf16. CE loss gets `logits.float()`.
- **Optimizers**: Muon for 2D hidden Linear weights (`adjust_lr_fn="match_rms_adamw"`), AdamW for embedding/head/biases/norms and non-2D params (`_ADAMW_ONLY_KEYWORDS` includes `phase_emb`; Muon rejects non-2D tensors).
- **Sampler shuffle**: `DistributedSampler` seed=0 + `set_epoch` gives a reproducible order; train_logkv resume skips consumed samples via a SkipSampler.
- **Legacy pipeline notes** (train_pipeline.py): Schedule1F1B loss collection via `losses=[]`, per-stage checkpoints + reconstructed `full_model.pt`, `STAGE_LAYER_SPLIT` for VRAM balance, per-step ReduceLROnPlateau on EMA loss. Cache building happens before `init_process_group` (sentinel file); all ranks barrier after checkpoint saves.

## Debugging Guidelines
- After modifying LogKV, run `uv run pytest test_logkv.py test_logkv_lm.py -v` before committing; the fp64 reference/step/predict equivalences must stay <1e-12.
- When modifying the data pipeline (packing, collation), add shape/length assertions; all packed sequences must be exactly context_length (`(seq + [PAD] * context_length)[:context_length]`).
- **NaN debugging**: training is deterministic with `seed=0` + `set_epoch(epoch)`, so resuming reproduces the same NaN at the same step. Loss=NaN with finite GradNorm typically means all-PAD labels (0/0 CE), not bad logits.
- Generation-quality checks use the repetition metric suite (temp 0.7/1.0 × 3 seeds × 1024 tokens: EOS rate, Q4 bigram distinct ratio, heavy-repetition count, longest same-char run) — see doc/logkv.md §6.6 for baselines.
- Legacy tests `test_step_split_consistency`/`test_predict_matches_forward` use loose tolerances (`atol=5e-3`) due to data-dependent xs propagation.

## Current Model Parameters (LogKV standard)
- d_model=1024, num_heads=8, d_ff=3072, chunk_size=4, num_layers=16 (~310M params)
- phase_emb=True (phase_levels=2), gated_attention=True, self_slot=True; decay fixed at log C
- context_length=2048, lr=2e-4 (linear warmup 1000), DDP batch 4/GPU × 6 GPUs
- mixed precision: fp32 master / bfloat16 autocast
- (Legacy pipeline model: d_model=2048, num_heads=16, d_ff=6144, num_layers=16, lr=5e-5)

## Datasets
Selected by `--dataset-type` in `train_logkv.py` / `train_pipeline.py`:
- `pretrain` (documents only):
  - `wikimedia/wikipedia` (20231101.ja, 20231101.en)
  - `hotchpotch/cc100-ja-documents`
  - `JeanKaddour/minipile`
- `instruct` (conversations only):
  - `shi3z/ja_conv_wikipedia_llama2pro8b_30k`
  - `shi3z/ja_conv_wikipedia_orion14B_100K`
  - `HuggingFaceH4/ultrachat_200k`
