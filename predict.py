"""
Text generation using a trained RecursiveCompressorLM model.

Usage:
    # From save_pretrained directory
    uv run python predict.py "吾輩は猫である。" --model-dir ./data/final_model

    # With options
    uv run python predict.py "吾輩は猫である。" --model-dir ./data/final_model \
        --context-length 256 --temperature 0.8 --top-p 0.9
"""

import argparse
import os
import torch
from transformers import AutoTokenizer

from configuration_recursive_compressor import RecursiveCompressorConfig
from dataset import TOKENIZER_NAME
from recursive_compressor_lm import RecursiveCompressorLM

torch.set_float32_matmul_precision("high")


_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


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


def _load_tokenizer(model_dir):
    if os.path.exists(os.path.join(model_dir, "tokenizer_config.json")):
        return AutoTokenizer.from_pretrained(model_dir)
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def predict(prompt, model_dir, context_length=1024, temperature=1.0, top_p=1.0, precision="bf16"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _load_model(model_dir, device, dtype=_DTYPES[precision])
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    tokenizer = _load_tokenizer(model_dir)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=context_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=False)


def main():
    parser = argparse.ArgumentParser(description="Generate text with RecursiveCompressorLM")
    parser.add_argument("prompt", type=str, help="入力テキスト")
    parser.add_argument("--model-dir", type=str, required=True, help="モデルディレクトリ (save_pretrained)")
    parser.add_argument("--context-length", type=int, default=1024, help="生成する最大コンテキスト長 (プロンプト含む合計トークン数)")
    parser.add_argument("--temperature", type=float, default=1.0, help="サンプリング温度")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p (nucleus) サンプリング閾値 (1.0で無効)")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16", help="推論精度")
    args = parser.parse_args()

    text = predict(args.prompt, args.model_dir, args.context_length, args.temperature, args.top_p, args.precision)
    print(text)


if __name__ == "__main__":
    main()
