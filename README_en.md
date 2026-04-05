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

```bash
# Single GPU
uv run python train.py

# Multi-GPU (e.g. 6 GPUs)
uv run torchrun --nproc_per_node=6 train.py
```

Training data is automatically downloaded from HuggingFace. Tokenized caches (numpy memmap) are stored in `$DATA_DIR/hf_cache/mmap/` and reused on subsequent runs.

#### Training Control

```bash
echo "pause"         > control.cmd   # Pause training
echo "resume"        > control.cmd   # Resume training
echo "save_and_exit" > control.cmd   # Save checkpoint and exit
```

Checkpoints are saved to `$DATA_DIR/checkpoints/` and automatically restored on restart.

### Text Generation

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
| `configuration_recursive_compressor.py` | Model config (extends PretrainedConfig) |
| `dataset.py` | HF dataset loading, tokenization, memmap caching |
| `train.py` | DDP training script (checkpointing, control commands) |
| `predict.py` | Text generation (loads model via `from_pretrained`) |
| `test_lm.py` | Tests |
| `.env.example` | Environment config example |

## Training Datasets

| Dataset | Type |
|---|---|
| `wikimedia/wikipedia` (ja, en) | Documents |
| `JeanKaddour/minipile` | Documents |
| `shi3z/ja_conv_wikipedia_llama2pro8b_30k` | Dialogue |
| `shi3z/ja_conv_wikipedia_orion14B_100K` | Dialogue |
| `HuggingFaceH4/ultrachat_200k` | Dialogue |

Document data is formatted with `[DOC]` markers, dialogue data with `[QUERY]`/`[ANSWER]` markers.

## Model Parameters

| Parameter | Value |
|---|---|
| d_model | 1024 |
| num_heads | 8 |
| d_ff | 2048 |
| chunk_size | 8 |
| compress_size | 4 |
| num_layers | 8 |
| context_length | 4096 |
| optimizer | RAdamScheduleFree |
| learning_rate | 3e-4 |
