"""
Text generation using a trained RecursiveCompressorLM model.

Usage:
    # From save_pretrained directory
    uv run python predict.py "吾輩は猫である。" --model-dir ./data/final_model

    # With options
    uv run python predict.py "吾輩は猫である。" --model-dir ./data/final_model \
        --context-length 256 --temperature 0.8
"""

import argparse
import os
import torch
from transformers import AutoTokenizer

from configuration_recursive_compressor import RecursiveCompressorConfig
from dataset import TOKENIZER_NAME
from recursive_compressor_lm import RecursiveCompressorLM

torch.set_float32_matmul_precision("high")


def sample_token(logits, temperature, top_p):
    """Sample one token from logits with temperature and top-p (nucleus) filtering.
    logits: (batch, vocab) — assumed fp32. Returns (batch,) long tensor."""
    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cumulative = sorted_probs.cumsum(dim=-1)
        # Keep tokens until cumulative prob >= top_p (always keep at least the top one)
        cutoff = cumulative > top_p
        cutoff[..., 0] = False
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        next_in_sorted = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_idx.gather(-1, next_in_sorted).squeeze(-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _load_model(model_dir, device, dtype=torch.bfloat16):
    """Load model from save_pretrained dir or pipeline checkpoint (full_model.pt).
    Casts to dtype on device. Defaults to bfloat16 to halve VRAM and avoid the
    autocast weight-cache that would otherwise duplicate fp32 weights."""
    full_model_pt = os.path.join(model_dir, "full_model.pt")

    if os.path.exists(full_model_pt):
        config = RecursiveCompressorConfig.from_pretrained(model_dir)
        model = RecursiveCompressorLM(config)
        state_dict = torch.load(full_model_pt, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        return model.to(dtype=dtype, device=device)
    else:
        return RecursiveCompressorLM.from_pretrained(model_dir).to(dtype=dtype, device=device)


_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def predict(prompt, model_dir, context_length=1024, temperature=1.0, top_p=1.0, precision="bf16"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _load_model(model_dir, device, dtype=_DTYPES[precision])
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    # Use tokenizer from model dir if available, otherwise fall back to default
    if os.path.exists(os.path.join(model_dir, "tokenizer_config.json")):
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    token_ids = tokenizer.encode(prompt)
    generated = list(token_ids)

    with torch.no_grad():
        # Feed prompt as a single sequence using step
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        logits, hidden = model.step(input_ids, None)
        logits = logits[:, -1, :]  # last position

        # Generate new tokens
        while len(generated) < context_length:
            next_token = sample_token(logits.float(), temperature, top_p)

            generated.append(next_token.item())
            if next_token.item() == tokenizer.eos_token_id:
                break

            next_input = next_token.unsqueeze(-1)  # (1, 1)
            logits, hidden = model.step(next_input, hidden)
            logits = logits[:, -1, :]

    return tokenizer.decode(generated)


def main():
    parser = argparse.ArgumentParser(description="Generate text with RecursiveCompressorLM")
    parser.add_argument("prompt", type=str, help="入力テキスト")
    parser.add_argument("--model-dir", type=str, required=True, help="モデルディレクトリ (save_pretrained)")
    parser.add_argument("--context-length", type=int, default=1024, help="生成する最大コンテキスト長")
    parser.add_argument("--temperature", type=float, default=1.0, help="サンプリング温度")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p (nucleus) サンプリング閾値 (1.0で無効)")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16", help="推論精度")
    args = parser.parse_args()

    text = predict(args.prompt, args.model_dir, args.context_length, args.temperature, args.top_p, args.precision)
    print(text)


if __name__ == "__main__":
    main()
