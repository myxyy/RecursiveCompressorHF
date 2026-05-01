English | [日本語](README.md)

# RecursiveCompressor

A language model implementation using a custom architecture with recursive compression.

## Architecture

RecursiveCompressor splits the input sequence into chunks, processes each chunk with causal attention (GatedAttention), and achieves inter-chunk information transfer through recursive compression and decompression.

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

### Training (Pipeline Parallel)

```bash
uv run torchrun --nproc_per_node=6 train_pipeline.py
```

The model is split across GPUs using pipeline parallelism. Trains in mixed precision (fp32 master weights + bfloat16 autocast) with a two-optimizer setup: Muon (2D hidden weights) + AdamW (embedding, head, biases, LayerNorms).

Training data is automatically downloaded from HuggingFace. Tokenized caches (numpy memmap) are stored in `$DATA_DIR/hf_cache/mmap/` and reused on subsequent runs.

#### Training Control

```bash
echo "pause"         > control.cmd   # Pause training
echo "resume"        > control.cmd   # Resume training
echo "save_and_exit" > control.cmd   # Save checkpoint and exit
```

Checkpoints are saved to `$DATA_DIR/checkpoints_pipeline/` and automatically restored on restart.

### Text Generation

```bash
# Single generation
uv run python predict.py "Once upon a time" --model-dir /path/to/checkpoint \
    --context-length 256 --temperature 0.8 --top-p 0.9

# Interactive streaming
uv run python predict_stream.py --model-dir /path/to/checkpoint \
    --context-length 1024 --temperature 0.8 --top-p 0.9
```

| Option | Description | Default |
|---|---|---|
| `prompt` | Input text (required for predict.py only) | - |
| `--model-dir` | Model directory (required) | - |
| `--context-length` | Maximum generation length in tokens | 1024 |
| `--temperature` | Sampling temperature | 1.0 |
| `--top-p` | top-p (nucleus) sampling threshold (1.0 disables) | 1.0 |

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
| `train_pipeline.py` | Pipeline parallel training (Muon + AdamW, bfloat16 autocast) |
| `predict.py` | Text generation |
| `predict_stream.py` | Interactive streaming generation |
| `test_lm.py` | Tests |
| `.env.example` | Environment config example |

## Training Datasets

Selected by `--dataset-type`:

### `pretrain` (documents)
| Dataset | Language |
|---|---|
| `wikimedia/wikipedia` (20231101.ja) | Japanese |
| `wikimedia/wikipedia` (20231101.en) | English |
| `hotchpotch/cc100-ja-documents` | Japanese |
| `JeanKaddour/minipile` | English |

### `instruct` (dialogue)
| Dataset | Language |
|---|---|
| `shi3z/ja_conv_wikipedia_llama2pro8b_30k` | Japanese |
| `shi3z/ja_conv_wikipedia_orion14B_100K` | Japanese |
| `HuggingFaceH4/ultrachat_200k` | English |

Documents are formatted as `<s>text` (no marker), dialogue as `<s>[QUERY]q[ANSWER]a`.
Long texts are split into context_length-sized chunks; short texts are packed together to minimize PAD waste.

## Model Parameters

| Parameter | Value |
|---|---|
| d_model | 2048 |
| num_heads | 16 |
| d_ff | 6144 |
| chunk_size | 4 |
| compress_size | 1 |
| num_layers | 16 |
| context_length | 2048 |
| optimizer | Muon (2D hidden) + AdamW (embedding/head/bias/norm) |
| learning rate | 5e-5 |
| precision | fp32 master weights + bfloat16 autocast |
