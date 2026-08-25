"""
Text generation from a LogKVLM checkpoint.

Usage:
    uv run python predict_logkv.py --model-dir $DATA_DIR/checkpoints_logkv/d1024-l16/checkpoint-20000/model \
        --max-new-tokens 1024 --temperature 0.7 --top-p 0.9 "日本の首都は"
    (省略時は訓練時と同じ3プロンプトで生成)
"""

import argparse

import torch
from dotenv import load_dotenv

from logkv_lm import LogKVLM
from dataset import get_tokenizer

DEFAULT_PROMPTS = ["日本の首都は", "昔々あるところに", "人工知能とは"]


def parse_args():
    p = argparse.ArgumentParser(description="LogKVLM text generation")
    p.add_argument("prompts", nargs="*", default=None)
    p.add_argument("--model-dir", type=str, required=True)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None, help="例: 0, cuda:3, cpu")
    return p.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    torch.manual_seed(args.seed)

    tokenizer = get_tokenizer()
    model = LogKVLM.from_pretrained(args.model_dir).to(device).eval()
    print(f"Loaded {args.model_dir} ({sum(p.numel() for p in model.parameters()):,} params) on {device}")

    prompts = args.prompts or DEFAULT_PROMPTS
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                             enabled=device.type == "cuda"):
            out = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=not args.greedy,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
        n_new = out.size(1) - input_ids.size(1)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"\n===== [{prompt}] ({n_new} new tokens) =====")
        print(text)


if __name__ == "__main__":
    main()
