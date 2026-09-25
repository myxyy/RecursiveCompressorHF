English | [日本語](README.md)

# RecursiveCompressor / LogKV

Code is organized into packages by purpose. Run the commands below from the repository root. See the [directory layout and migration guide](doc/repository-layout.md) for old paths and historical experiment reproduction.

See the [experiment index](doc/logkv-experiments.md) for comparisons, evaluation artifacts and reproduction commits (in Japanese).

The standard training configuration now uses width4 CausalConv before each block attention, with phase embeddings off and gate/self slot on. These are the new-training CLI defaults; use `--conv-kernel-size 0`, `--no-gated-attention` and `--no-self-slot` for ablations.
The fixed10 Copying checkpoint achieved 256/256 exact at all 41 tested horizons through T131072 and 8/8 at T16777216. See the [CausalConv report](doc/logkv-causal-conv.md).

A language model implementation of **LogKV**, a custom architecture based on hierarchical kv compression.

![LogKV kv-cache structure (without overlap)](logkv-refine.drawio.png)

## Architecture (LogKV)

LogKV recursively compresses the sequence chunk by chunk (C = chunk_size) with attention pooling. Every query attends through a **single softmax to completed sub-units in its current chunk at each level**. These intervals partition the entire past without gaps or overlap, using at most C−1 kv slots per level. For C=4, position 4 sees only summary 0–3; position 5 sees summary 0–3 and token 4.

- The receptive field covers the whole sequence with only O(C·log L) kv entries per attention
- The sequential-inference hidden state is also O(C·log L·d), logarithmic in sequence length; only the unfinished chunk is retained at each level
- `forward` / `step` (arbitrary-length chunked processing with hidden-state carry-over) / `predict` (single token) are implemented and tested to agree to machine precision in fp64
- `predict` uses a dedicated single-token path that reads cached slots directly and compresses only completed chunks. Cached decoding in HF `generate()` uses it automatically. See [timing and numerical checks](doc/logkv-fast-predict.md).

The standard configuration consists of (see [doc/logkv.md](doc/logkv.md) for the experimental record, in Japanese):

| Component | Description |
|---|---|
| Level decay | Logit bias −i·log C for level-i slots. Retains the original coarse-level penalty; its effectiveness after overlap removal needs evaluation |
| CausalConv | Width4 depthwise causal convolution residual before each block attention; phase embeddings are disabled |
| Multi-head | Heads folded into the batch dimension |
| Gated attention | Per-head attention output multiplied by sigmoid(W_g x) |
| Self slot | One extra slot holding the query token's own k/v (same semantics as a standard causal mask); gives the softmax an "attend to nothing" option and stabilizes gradients |

The language model (LogKVLM) is `Embedding → LogKVBlock × num_layers → RMSNorm → Linear`, extending HuggingFace's `PreTrainedModel` (`save_pretrained` / `from_pretrained` / `generate`).

**In the earlier experiment with two-level phase embeddings and fixed Copying memory length M=10, the refined layout achieved 100% exact-match Copying accuracy at all 41 evaluated horizons through T=131,072 after training through T=2,028** (256 samples per horizon, both best and final checkpoints). An additional probe using that final checkpoint at T=16,777,216 also copied all 8 samples correctly. This is a historical result for that configuration, not an achievement of the alternative positional encodings on the experimental branches. Selective Copying accuracy decreases against the original layout under matched conditions ([experiment details](doc/logkv-refine-experiments.md), one training seed per condition). LM generation quality with the refined layout remains unevaluated. Existing checkpoint weights can be loaded, but outputs change under the new layout.

As of 2026-09-13, positional-encoding tuning is paused and training will use the existing main architecture.
Only documentation, results, and experiment code have been imported; model code, configuration, and training CLIs are unchanged.
See the [experiment index and reproduction instructions](doc/logkv-experiments.md), including
[Copying](doc/logkv-aligned-rope.md) and [Selective Copying](doc/logkv-aligned-selective.md).
Experimental positional-encoding flags are not available on main.

## Setup

```bash
uv sync
cp .env.example .env
# Edit DATA_DIR in .env (storage for datasets and checkpoints)
```

## Usage

### Training (DDP data parallel)

```bash
uv run torchrun --nproc_per_node=6 --module training.train_logkv \
    --run-name myrun --conv-kernel-size 4 --gated-attention --self-slot
```

Trains in mixed precision (fp32 master weights + bfloat16 autocast) with a two-optimizer setup: Muon (2D hidden weights) + AdamW. The attention pass uses online softmax + activation checkpointing to reduce VRAM.

Training data is automatically downloaded from HuggingFace; tokenized caches (numpy memmap) are stored in `$DATA_DIR/hf_cache/mmap/ctx{context_length}/`. Checkpoints go to `$DATA_DIR/checkpoints_logkv/{run-name}/` and can be resumed with `--resume latest` (already-consumed data is skipped; `--max-steps` is an absolute step count). Every 1000 steps, sample generations from Japanese prompts are appended to `samples.log`.

#### Training control

```bash
just pause          # Pause (process stays alive, GPUs idle)
just resume         # Resume
just save-and-exit  # Save a checkpoint and exit -> resume with --resume latest
```

### Pipeline-parallel training

Use `training/train_logkv_pipeline.py` to split one LogKV model across GPUs:

```bash
uv run torchrun --standalone --nproc_per_node=6 --module training.train_logkv_pipeline \
    --run-name pipeline-base --batch-size 12 --n-microbatches 12 --grad-accum 2 \
    --stage-layer-split 2,2,3,3,3,3
```

The effective batch here is 24, without multiplying by GPU count. Each checkpoint exports
`$DATA_DIR/checkpoints_logkv_pipeline/pipeline-base/checkpoint-{step}/model` in standard LogKV HF format,
including the tokenizer. Pass that directory to `python -m inference.predict_stream --model-dir`.
See [pipeline training, resume and memory requirements](doc/logkv-pipeline.md) (Japanese).

### Text generation

```bash
# One-shot generation
uv run python -m inference.predict_logkv --model-dir $DATA_DIR/checkpoints_logkv/myrun/checkpoint-5000/model \
    --max-new-tokens 1024 --temperature 0.7 --top-p 0.9 "日本の首都は"

# Interactive streaming (architecture auto-detected from config.json)
uv run python -m inference.predict_stream --model-dir /path/to/checkpoint \
    --context-length 4096 --temperature 0.7 --top-p 0.9
```

### Tests and basic experiments

```bash
uv run pytest tests/logkv/test_logkv.py tests/logkv/test_logkv_lm.py -v   # LogKV (incl. fp64 machine-precision equivalence)
uv run pytest tests/legacy/test_lm.py -v                       # legacy architecture

# Copy Memory Problem / Selective Copying (long-range memory benchmarks)
uv run python -m exp.copying.train --arch logkv --conv-kernel-size 4 --gated-attention --self-slot \
    --run-name myrun --t-dist loguniform
uv run python -m exp.copying.evaluate --run-name myrun --max-t-exp 17
```

## Files

| File | Description |
|---|---|
| `models/logkv/attention.py` | LogKV architecture (`forward`/`step`/`predict`, LogKVBlock) |
| `models/logkv/modeling.py` | Language model LogKVLM (extends PreTrainedModel) |
| `models/logkv/configuration.py` | Model config (extends PretrainedConfig) |
| `training/train_logkv.py` | DDP data-parallel training (Muon + AdamW, bfloat16 autocast) |
| `inference/predict_logkv.py` | Text generation (LogKV) |
| `inference/predict.py` / `inference/predict_stream.py` | Generation / interactive streaming (auto-detects old vs new architecture) |
| `data_pipeline/dataset.py` | HF dataset loading, tokenization, memmap caching |
| `tests/logkv/test_logkv.py` / `tests/logkv/test_logkv_lm.py` | LogKV tests |
| `exp/copying/`, `exp/selective_copying/` | Long-range memory experiment suites |
| `doc/logkv.md` | Design and experimental findings for LogKV (Japanese) |
| `models/logkv/pipeline.py` / `training/train_logkv_pipeline.py` | LogKV pipeline parallelism |
| `models/recursive_compressor/` / `training/train_pipeline.py` | Legacy architecture and training |
| `tests/` / `benchmarks/` | Tests and inference benchmarks |
| `.env.example` | Environment template |

### Legacy architecture (RecursiveCompressor)

The previous implementation — inter-chunk information transfer through recursive compression/decompression — is kept: `models/recursive_compressor/attention.py` / `models/recursive_compressor/modeling.py` / `models/recursive_compressor/pipeline.py` / `models/recursive_compressor/configuration.py` / `training/train_pipeline.py` (6-GPU pipeline parallel: `uv run torchrun --nproc_per_node=6 --module training.train_pipeline`). See [doc/copying-memory-branch-changes.md](doc/copying-memory-branch-changes.md) for its history (Japanese).

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
| phase_emb | off |
| conv_kernel_size | 4 |
| self_slot | on |
| gated_attention | on |
| optimizer | Muon (2D hidden) + AdamW (embedding/head/bias/norm/phase emb) |
| learning rate | 2e-4 (linear warmup 1000) |
| precision | fp32 master weights + bfloat16 autocast |

For checkpoint compatibility, low-level configuration defaults remain unchanged. When constructing `LogKVConfig` directly, specify `conv_kernel_size=4, gated_attention=True, self_slot=True, phase_emb=False` for the standard model.
