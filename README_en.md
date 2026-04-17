English | [日本語](README.md)

# RecursiveCompressor

A language model implementation using a custom architecture with recursive compression.

## Architecture

RecursiveCompressor splits the input sequence into chunks, processes each chunk with causal attention, and achieves inter-chunk information transfer through recursive compression and decompression.

The language model (RecursiveCompressorLM) has the following structure:

```
Embedding → RecursiveCompressor × num_layers → LayerNorm → Linear
```

It extends HuggingFace's `PreTrainedModel`, supporting `save_pretrained` / `from_pretrained` / `push_to_hub`.

## Setup

```bash
uv sync
cp .env.example .env
# Edit DATA_DIR in .env (storage for datasets and checkpoints)
```

## Usage

### Training

Two parallelism modes are supported.

```bash
# Single GPU
uv run python train.py

# DDP (data parallel, 6 GPUs)
uv run torchrun --nproc_per_node=6 train.py

# Pipeline parallel (model split across GPUs, 6 GPUs)
uv run torchrun --nproc_per_node=6 train_pipeline.py
```

| Mode | Use case |
|---|---|
| DDP | Faster training when the model fits on a single GPU |
| Pipeline | When the model is too large for a single GPU |

Training data is automatically downloaded from HuggingFace. Tokenized caches (numpy memmap) are stored in `$DATA_DIR/hf_cache/mmap/` and reused on subsequent runs.

#### Training Control

```bash
echo "pause"         > control.cmd   # Pause training
echo "resume"        > control.cmd   # Resume training
echo "save_and_exit" > control.cmd   # Save checkpoint and exit
```

Checkpoints are saved to `$DATA_DIR/checkpoints/` (DDP) or `$DATA_DIR/checkpoints_pipeline/` (Pipeline) and automatically restored on restart.

### Text Generation

Both DDP and pipeline checkpoints work with the same command.

```bash
uv run python predict.py "Once upon a time" --model-dir ./data/final_model \
    --context-length 256 --temperature 0.8
```

| Option | Description | Default |
|---|---|---|
| `prompt` | Input text (required) | - |
| `--model-dir` | Model directory (required) | - |
| `--context-length` | Maximum generation length in tokens | 1024 |
| `--temperature` | Sampling temperature | 1.0 |

### Tests

```bash
uv run pytest test_lm.py -v
```

## File Structure

| File | Description |
|---|---|
| `recursive_compressor.py` | RecursiveCompressor module (`step`/`forward`/`predict`) |
| `recursive_compressor_lm.py` | Language model (extends PreTrainedModel) |
| `recursive_compressor_lm_pipeline.py` | Pipeline parallel stage wrapper |
| `configuration_recursive_compressor.py` | Model config (extends PretrainedConfig) |
| `dataset.py` | HF dataset loading, tokenization, memmap caching |
| `train.py` | DDP training script (checkpointing, control commands) |
| `train_pipeline.py` | Pipeline parallel training script |
| `predict.py` | Text generation (supports both DDP and pipeline checkpoints) |
| `test_lm.py` | Tests |
| `.env.example` | Environment config example |

## Training Datasets (Japanese only)

| Dataset | Type |
|---|---|
| `wikimedia/wikipedia` (20231101.ja) | Documents |
| `hotchpotch/cc100-ja-documents` | Documents |
| `shi3z/ja_conv_wikipedia_llama2pro8b_30k` | Dialogue |
| `shi3z/ja_conv_wikipedia_orion14B_100K` | Dialogue |

Document data is formatted with `[DOC]` markers, dialogue data with `[QUERY]`/`[ANSWER]` markers. Short documents are packed into single samples to reduce PAD waste.

## Model Parameters

| Parameter | Value |
|---|---|
| d_model | 2048 |
| num_heads | 16 |
| d_ff | 4096 |
| chunk_size | 4 |
| compress_size | 1 |
| num_layers | 12 (DDP) / 16 (Pipeline) |
| context_length | 2048 |
| optimizer | RAdamScheduleFree |
| dtype | float32 |
