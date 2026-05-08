"""
Streaming text generation with RecursiveCompressorLM.

Usage:
    uv run python predict_stream.py --model-dir /path/to/checkpoint
    uv run python predict_stream.py --model-dir /path/to/checkpoint \
        --context-length 1024 --temperature 0.8

Reads prompts interactively from stdin and prints generated tokens
as they are produced. Reuses hidden state across the prompt.
Type 'exit' to quit, or 'reset' to clear the in-progress hidden state.
"""

import argparse
import os
import sys
import time
import torch
from transformers import AutoTokenizer, TextStreamer

from configuration_recursive_compressor import RecursiveCompressorConfig
from dataset import TOKENIZER_NAME
from predict import _DTYPES, sample_token
from recursive_compressor_lm import RecursiveCompressorLM

torch.set_float32_matmul_precision("high")


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


def stream_generate(model, tokenizer, prompt, context_length, temperature, top_p, device):
    """Generate tokens one at a time, streaming output to stdout.
    Returns (num_generated, elapsed_seconds)."""
    streamer = TextStreamer(tokenizer, skip_prompt=False, skip_special_tokens=True)

    token_ids = tokenizer.encode(prompt)

    # Stream the prompt itself
    streamer.put(torch.tensor(token_ids))

    with torch.no_grad():
        # Feed prompt as a single sequence
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        logits, hidden = model.step(input_ids, None)
        logits = logits[:, -1, :]

        num_generated = 0
        start_time = time.time()

        while len(token_ids) + num_generated < context_length:
            next_token = sample_token(logits.float(), temperature, top_p)

            tok_id = next_token.item()
            streamer.put(next_token.cpu())
            num_generated += 1

            if tok_id == tokenizer.eos_token_id:
                break

            next_input = next_token.unsqueeze(-1)
            logits, hidden = model.step(next_input, hidden)
            logits = logits[:, -1, :]

        streamer.end()
        elapsed = time.time() - start_time

    return num_generated, elapsed


def main():
    parser = argparse.ArgumentParser(description="Stream text generation with RecursiveCompressorLM")
    parser.add_argument("--model-dir", type=str, required=True, help="モデルディレクトリ")
    parser.add_argument("--context-length", type=int, default=1024, help="生成する最大コンテキスト長")
    parser.add_argument("--temperature", type=float, default=1.0, help="サンプリング温度")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p (nucleus) サンプリング閾値 (1.0で無効)")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16", help="推論精度")
    parser.add_argument("--device", type=str, default=None,
                        help="使用デバイス。例: 0, cuda:3, cpu。未指定なら自動 (cuda:0 / cpu)")
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        # Allow shortcut like "0", "1" to mean "cuda:0", "cuda:1"
        spec = args.device
        if spec.isdigit():
            spec = f"cuda:{spec}"
        device = torch.device(spec)

    print("Loading model...", flush=True)
    model = _load_model(args.model_dir, device, dtype=_DTYPES[args.precision])
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    tokenizer = _load_tokenizer(args.model_dir)

    state = {
        "context_length": args.context_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    print(f"Device: {device}, precision: {args.precision}, context_length: {state['context_length']}, "
          f"temperature: {state['temperature']}, top_p: {state['top_p']}")
    print("Commands: 'exit', 'temperature [val]', 'top-p [val]', 'context-length [val]'")

    # Map command name -> (state key, value parser)
    commands = {
        "temperature": ("temperature", float),
        "top-p": ("top_p", float),
        "context-length": ("context_length", int),
    }

    while True:
        try:
            prompt = input("\n>>> ")
        except EOFError:
            break
        stripped = prompt.strip()
        if stripped.lower() == "exit":
            break
        if not stripped:
            continue

        # Check if input is a command
        parts = stripped.split(maxsplit=1)
        cmd = parts[0].lower()
        if cmd in commands:
            key, parse = commands[cmd]
            if len(parts) == 1:
                print(f"{cmd} = {state[key]}")
            else:
                try:
                    state[key] = parse(parts[1])
                    print(f"{cmd} set to {state[key]}")
                except ValueError:
                    print(f"Invalid value for {cmd}: {parts[1]!r}")
            continue

        num_generated, elapsed = stream_generate(
            model, tokenizer, prompt,
            state["context_length"], state["temperature"], state["top_p"], device,
        )
        if elapsed > 0 and num_generated > 0:
            print(f"\n[{num_generated} tokens, {elapsed:.2f}s, {num_generated / elapsed:.2f} tok/s]")


if __name__ == "__main__":
    main()
