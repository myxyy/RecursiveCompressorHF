English | [日本語](README.md)

# RecursiveCompressor

A language model implementation using a custom architecture with recursive compression.

## Architecture

RecursiveCompressor splits the input sequence into chunks, processes each chunk with causal attention, and achieves inter-chunk information transfer through recursive compression and decompression.

The language model (RecursiveCompressorLM) has the following structure:

```
Embedding → RecursiveCompressor × num_layers → LayerNorm → Linear
```

## Setup

```bash
uv sync
```

## Usage

### Training

Place `.txt` training files in the `text/` directory and run:

```bash
uv run python train.py
```

Uses the `elyza/ELYZA-japanese-Llama-2-7b-fast` tokenizer.

### Text Generation

```bash
# Fast (reuses hidden state, generates one token at a time)
uv run python predict_fast.py "吾輩は猫である。" --context-length 256 --temperature 0.8

# Naive (full forward pass each step)
uv run python predict_naive.py "吾輩は猫である。" --context-length 256 --temperature 0.8
```

| Option | Description | Default |
|---|---|---|
| `prompt` | Input text (required) | - |
| `--context-length` | Maximum generation length in tokens | 1024 |
| `--temperature` | Sampling temperature | 1.0 |
| `--weights` | Path to weights file | `recursive_compressor_lm.pth` |

### Tests

```bash
uv run pytest test_lm.py -v
```

## File Structure

| File | Description |
|---|---|
| `recursive_compressor.py` | RecursiveCompressor module |
| `recursive_compressor_lm.py` | Language model (RecursiveCompressorLM) |
| `dataset.py` | Text dataset (TextDataset) |
| `train.py` | Training script |
| `predict_fast.py` | Fast text generation (hidden state reuse) |
| `predict_naive.py` | Naive text generation (full forward each step) |
| `test_lm.py` | Tests |

## Default Hyperparameters

| Parameter | Value |
|---|---|
| d_model | 512 |
| num_heads | 8 |
| d_ff | 2048 |
| chunk_size | 8 |
| compress_size | 4 |
| num_layers | 4 |
| context_length | 2048 |
| batch_size | 4 |
| learning_rate | 3e-4 |
