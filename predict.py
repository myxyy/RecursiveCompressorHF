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


def _load_model(model_dir, device):
    """Load model from save_pretrained dir or pipeline checkpoint (full_model.pt)."""
    full_model_pt = os.path.join(model_dir, "full_model.pt")
    config_json = os.path.join(model_dir, "config.json")

    if os.path.exists(full_model_pt):
        # Pipeline checkpoint: load config.json + full_model.pt
        config = RecursiveCompressorConfig.from_pretrained(model_dir)
        model = RecursiveCompressorLM(config)
        state_dict = torch.load(full_model_pt, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        return model.to(device)
    else:
        # Standard save_pretrained directory
        return RecursiveCompressorLM.from_pretrained(model_dir).to(device)


def predict(prompt, model_dir, context_length=1024, temperature=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _load_model(model_dir, device)
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
            next_logits = logits / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

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
    args = parser.parse_args()

    text = predict(args.prompt, args.model_dir, args.context_length, args.temperature)
    print(text)


if __name__ == "__main__":
    main()
