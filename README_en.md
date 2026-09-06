English | [日本語](README.md)

# RecursiveCompressor / LogKV

A language model implementation of **LogKV**, a custom architecture based on hierarchical kv compression.

![LogKV kv-cache structure (without overlap)](logkv-refine.drawio.png)

## Architecture (LogKV)

LogKV recursively compresses the sequence chunk by chunk (C = chunk_size) with attention pooling. Every query attends through a **single softmax to completed sub-units in its current chunk at each level**. These intervals partition the entire past without gaps or overlap, using at most C−1 kv slots per level. For C=4, position 4 sees only summary 0–3; position 5 sees summary 0–3 and token 4.

- The receptive field covers the whole sequence with only O(C·log L) kv entries per attention
- The sequential-inference hidden state is also O(C·log L·d), logarithmic in sequence length; only the unfinished chunk is retained at each level
- `forward` / `step` (arbitrary-length chunked processing with hidden-state carry-over) / `predict` (single token) are implemented and tested to agree to machine precision in fp64

The standard configuration consists of (see [doc/logkv.md](doc/logkv.md) for the experimental record, in Japanese):

| Component | Description |
|---|---|
| Level decay | Logit bias −i·log C for level-i slots. Retains the original coarse-level penalty; its effectiveness after overlap removal needs evaluation |
| Phase embedding | Learned vectors for the low base-C digits of the position (period 16). Fixes positional degeneracy inside runs of identical tokens |
| Multi-head | Heads folded into the batch dimension |
| Gated attention | Per-head attention output multiplied by sigmoid(W_g x) |
| Self slot | One extra slot holding the query token's own k/v (same semantics as a standard causal mask); gives the softmax an "attend to nothing" option and stabilizes gradients |

The language model (LogKVLM) is `Embedding → LogKVBlock × num_layers → RMSNorm → Linear`, extending HuggingFace's `PreTrainedModel` (`save_pretrained` / `from_pretrained` / `generate`).

**With the original overlapping layout, a model trained with horizon 2,028 tokens performs perfect copying from 16.7M tokens away (8,273× the training horizon)** ([doc/logkv.md](doc/logkv.md) §6.13). Training and generation quality with the refined layout have not yet been evaluated (§6.17). Existing checkpoint weights can be loaded, but outputs change under the new layout.

## Setup

```bash
uv sync
cp .env.example .env
# Edit DATA_DIR in .env (storage for datasets and checkpoints)
```

## Usage

### Training (DDP data parallel)

```bash
uv run torchrun --nproc_per_node=6 train_logkv.py \
    --run-name myrun --phase-emb --phase-levels 2 --gated-attention --self-slot
```

Trains in mixed precision (fp32 master weights + bfloat16 autocast) with a two-optimizer setup: Muon (2D hidden weights) + AdamW. The attention pass uses online softmax + activation checkpointing to reduce VRAM.

Training data is automatically downloaded from HuggingFace; tokenized caches (numpy memmap) are stored in `$DATA_DIR/hf_cache/mmap/ctx{context_length}/`. Checkpoints go to `$DATA_DIR/checkpoints_logkv/{run-name}/` and can be resumed with `--resume latest` (already-consumed data is skipped; `--max-steps` is an absolute step count). Every 1000 steps, sample generations from Japanese prompts are appended to `samples.log`.

#### Training control

```bash
just pause          # Pause (process stays alive, GPUs idle)
just resume         # Resume
just save-and-exit  # Save a checkpoint and exit -> resume with --resume latest
```

### Text generation

```bash
# One-shot generation
uv run python predict_logkv.py --model-dir $DATA_DIR/checkpoints_logkv/myrun/checkpoint-5000/model \
    --max-new-tokens 1024 --temperature 0.7 --top-p 0.9 "日本の首都は"

# Interactive streaming (architecture auto-detected from config.json)
uv run python predict_stream.py --model-dir /path/to/checkpoint \
    --context-length 4096 --temperature 0.7 --top-p 0.9
```

### Tests and basic experiments

```bash
uv run pytest test_logkv.py test_logkv_lm.py -v   # LogKV (incl. fp64 machine-precision equivalence)
uv run pytest test_lm.py -v                       # legacy architecture

# Copy Memory Problem / Selective Copying (long-range memory benchmarks)
uv run python exp/copying/train.py --arch logkv --phase-emb --phase-levels 2 --gated-attention --self-slot \
    --run-name myrun --t-dist loguniform
uv run python exp/copying/evaluate.py --run-name myrun --max-t-exp 17
```

## Files

| File | Description |
|---|---|
| `logkv.py` | LogKV architecture (`forward`/`step`/`predict`, LogKVBlock) |
| `logkv_lm.py` | Language model LogKVLM (extends PreTrainedModel) |
| `configuration_logkv.py` | Model config (extends PretrainedConfig) |
| `train_logkv.py` | DDP data-parallel training (Muon + AdamW, bfloat16 autocast) |
| `predict_logkv.py` | Text generation (LogKV) |
| `predict.py` / `predict_stream.py` | Generation / interactive streaming (auto-detects old vs new architecture) |
| `dataset.py` | HF dataset loading, tokenization, memmap caching |
| `test_logkv.py` / `test_logkv_lm.py` | LogKV tests |
| `exp/copying/`, `exp/selective-copying/` | Long-range memory experiment suites |
| `doc/logkv.md` | Design and experimental findings for LogKV (Japanese) |
| `.env.example` | Environment template |

### Legacy architecture (RecursiveCompressor)

The previous implementation — inter-chunk information transfer through recursive compression/decompression — is kept: `recursive_compressor.py` / `recursive_compressor_lm.py` / `recursive_compressor_lm_pipeline.py` / `configuration_recursive_compressor.py` / `train_pipeline.py` (6-GPU pipeline parallel: `uv run torchrun --nproc_per_node=6 train_pipeline.py`). See [doc/copying-memory-branch-changes.md](doc/copying-memory-branch-changes.md) for its history (Japanese).

## Training datasets

Selected with `--dataset-type`:

### `pretrain` (documents)
| Dataset | Language |
|---|---|
| `wikimedia/wikipedia` (20231101.ja) | Japanese |
| `wikimedia/wikipedia` (20231101.en) | English |
| `hotchpotch/cc100-ja-documents` | Japanese |
| `JeanKaddour/minipile` | English |

### `instruct` (conversations)
| Dataset | Language |
|---|---|
| `shi3z/ja_conv_wikipedia_llama2pro8b_30k` | Japanese |
| `shi3z/ja_conv_wikipedia_orion14B_100K` | Japanese |
| `HuggingFaceH4/ultrachat_200k` | English |

The data format is Llama-2 style: documents are `<s>text</s>`; conversations are `<s>[INST]q[/INST]a</s>` per turn. Long texts are split into context_length chunks and short ones are packed together to reduce PAD waste (no cross-conversation packing; answer-only loss for instruct data).

## Model parameters (LogKV standard configuration)

| Parameter | Value |
|---|---|
| d_model | 1024 |
| num_heads | 8 |
| d_ff | 3072 |
| chunk_size | 4 |
| num_layers | 16 |
| context_length | 2048 |
| phase_emb / phase_levels | on / 2 (period 16) |
| gated_attention | on |
| optimizer | Muon (2D hidden) + AdamW (embedding/head/bias/norm/phase emb) |
| learning rate | 2e-4 (linear warmup 1000) |
| precision | fp32 master weights + bfloat16 autocast |
