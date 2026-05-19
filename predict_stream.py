"""
Streaming text generation with RecursiveCompressorLM.

Usage:
    uv run python predict_stream.py --model-dir /path/to/checkpoint
    uv run python predict_stream.py --model-dir /path/to/checkpoint \
        --context-length 1024 --temperature 0.8 --top-p 0.9

Reads prompts interactively from stdin and prints generated tokens
as they are produced.

Commands at the prompt:
    exit                     - quit
    temperature [val]        - show or set temperature
    top-p [val]              - show or set top_p
    context-length [val]     - show or set max total token length

Input editing:
    Enter                    - submit
    Alt+Enter (Esc, Enter)   - insert newline (multi-line input)
"""

import argparse
import time
import torch
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from transformers import TextStreamer

from predict import _DTYPES, _load_model, _load_tokenizer


def _make_prompt_session():
    """Multi-line REPL. Enter submits; Alt+Enter (Esc then Enter) inserts a newline.

    Terminals can't distinguish Shift+Enter from Enter at the byte level, so to use
    Shift+Enter as a newline, configure the terminal to send Esc+Enter (\\x1b\\r)
    for Shift+Enter. In VSCode, add to keybindings.json:
        {"key": "shift+enter", "command": "workbench.action.terminal.sendSequence",
         "args": {"text": "\\u001b\\r"}, "when": "terminalFocus"}
    """
    bindings = KeyBindings()

    @bindings.add("enter")
    def _(event):
        event.current_buffer.validate_and_handle()

    @bindings.add("escape", "enter")
    def _(event):
        event.current_buffer.insert_text("\n")

    return PromptSession(key_bindings=bindings, multiline=True)

torch.set_float32_matmul_precision("high")


def stream_generate(model, tokenizer, prompt, context_length, temperature, top_p, device):
    """Run generate() with TextStreamer. Returns (num_generated, elapsed_seconds)."""
    streamer = TextStreamer(tokenizer, skip_prompt=False, skip_special_tokens=True)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.size(1)

    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=context_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            streamer=streamer,
        )
    elapsed = time.time() - start_time
    num_generated = output_ids.size(1) - prompt_len
    return num_generated, elapsed


def main():
    parser = argparse.ArgumentParser(description="Stream text generation with RecursiveCompressorLM")
    parser.add_argument("--model-dir", type=str, required=True, help="モデルディレクトリ")
    parser.add_argument("--context-length", type=int, default=1024, help="生成する最大コンテキスト長 (プロンプト含む合計)")
    parser.add_argument("--temperature", type=float, default=1.0, help="サンプリング温度")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p (nucleus) サンプリング閾値 (1.0で無効)")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16", help="推論精度")
    parser.add_argument("--device", type=str, default=None,
                        help="使用デバイス。例: 0, cuda:3, cpu。未指定なら自動 (cuda:0 / cpu)")
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
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
    print("Input: Enter to submit, Alt+Enter (or Esc then Enter) for newline")

    commands = {
        "temperature": ("temperature", float),
        "top-p": ("top_p", float),
        "context-length": ("context_length", int),
    }

    session = _make_prompt_session()
    while True:
        try:
            prompt = session.prompt("\n>>> ")
        except (EOFError, KeyboardInterrupt):
            break
        stripped = prompt.strip()
        if stripped.lower() == "exit":
            break
        if not stripped:
            continue

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
