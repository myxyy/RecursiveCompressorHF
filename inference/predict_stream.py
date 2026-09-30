"""
Streaming text generation with RecursiveCompressorLM / LogKVLM (auto-detected from config.json).

Usage:
    uv run python -m inference.predict_stream --model-dir /path/to/checkpoint
    uv run python -m inference.predict_stream --model-dir /path/to/checkpoint \
        --context-length 1024 --temperature 0.8 --top-p 0.9

Reads prompts interactively from stdin and prints generated tokens
as they are produced. A terminal status bar shows generated tokens and average
tok/s (including prefill), reset per prompt. Disable with --no-status-bar.

Commands at the prompt:
    exit                     - quit
    temperature [val]        - show or set temperature
    top-p [val]              - show or set top_p
    context-length [val]     - show or set max total token length
    skip-special-tokens [true/false] - show or set whether to skip special tokens in output
    stop-on-eos [true/false] - show or set whether to stop generation on EOS token
    ignore-eos [on/off]      - exclude EOS tokens from generation (default off)
    penalty-add [val]        - repetition penalty increment (0..2, default 0 = off)
    penalty-decay [val]      - retained penalty per generated token (0..1, default 0)

Input editing:
    Enter                    - submit
    Alt+Enter (Esc, Enter)   - insert newline (multi-line input)
    Ctrl+C (during gen)      - interrupt generation and return to prompt
"""

import argparse
import os
import signal
import sys
import time
import torch
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList, TextStreamer

from inference.predict import _DTYPES, _load_model, _load_tokenizer


class _InterruptStoppingCriteria(StoppingCriteria):
    """Halts generation when `interrupted` is flipped to True (e.g. by SIGINT)."""
    def __init__(self):
        self.interrupted = False

    def __call__(self, input_ids, scores, **kwargs):
        return self.interrupted


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


class _GenerationStatus:
    """Reserve the terminal's last row while streaming into its scroll region."""

    def __init__(self, enabled=True):
        self.output = sys.stdout
        self.enabled = enabled and self.output.isatty() and os.environ.get("TERM") != "dumb"
        self.size = None
        self.tokens = 0
        self.started = time.perf_counter()
        self.last_refresh = float("-inf")

    def _resize(self):
        try:
            size = os.get_terminal_size(self.output.fileno())
        except (OSError, ValueError):
            return False
        if size.lines < 3 or size.columns < 2:
            return False
        if size != self.size:
            # Changing the scroll region homes the cursor. Resume above the bar.
            self.output.write(f"\033[1;{size.lines - 1}r\033[{size.lines - 1};1H")
            self.size = size
        return True

    def refresh(self, force=False):
        now = time.perf_counter()
        if not self.enabled or (not force and now - self.last_refresh < 0.1):
            return
        if not self._resize():
            return
        elapsed = now - self.started
        rate = self.tokens / elapsed if elapsed > 0 else 0.0
        label = f" {self.tokens:,} tokens | {rate:.2f} tok/s | {elapsed:.1f}s "
        # ASCII label; leave the last column unused to avoid automatic wrapping.
        label = label[:self.size.columns - 1].ljust(self.size.columns - 1)
        self.output.write(f"\0337\033[{self.size.lines};1H\033[2K\033[7m{label}\033[0m\0338")
        self.output.flush()
        self.last_refresh = now

    def write(self, text):
        if self.enabled:
            self._resize()
        self.output.write(text)
        self.output.flush()

    def close(self):
        if self.size is not None:
            # Restore normal scrolling even if generate raises or is interrupted.
            self.output.write(f"\0337\033[r\033[{self.size.lines};1H\033[2K\0338")
            self.output.flush()
            self.size = None


class _StatusTextStreamer(TextStreamer):
    """Count generated token IDs, including hidden special tokens, not text chunks."""

    def __init__(self, tokenizer, status, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.status = status
        self._received_prompt = False

    def put(self, value):
        is_prompt = not self._received_prompt
        super().put(value)
        self._received_prompt = True
        if not is_prompt:
            self.status.tokens += value.numel()
        self.status.refresh()

    def on_finalized_text(self, text, stream_end=False):
        self.status.write(text + ("\n" if stream_end else ""))


class _DecayingRepetitionPenalty(LogitsProcessor):
    """Subtract a generated-token-only, exponentially decaying FP32 penalty.

    HF calls this before temperature/top-p processing. Consume newly generated
    IDs on the next call; the prompt never contributes. Each generation owns a
    fresh instance, since predict_stream does not retain conversation state.
    """

    def __init__(self, prompt_length, penalty_add, penalty_decay, eos_ids):
        self.processed = prompt_length
        self.add = penalty_add
        self.decay = penalty_decay
        self.eos_ids = list(eos_ids)
        self.penalty = None

    def __call__(self, input_ids, scores):
        if self.penalty is None:
            self.penalty = torch.zeros_like(scores, dtype=torch.float32)
        for position in range(self.processed, input_ids.shape[1]):
            self.penalty.mul_(self.decay)
            token = input_ids[:, position:position + 1]
            self.penalty.scatter_add_(1, token, torch.full_like(token, self.add, dtype=torch.float32))
            # EOS can still stop generation; ignore-eos separately excludes it.
            self.penalty[:, self.eos_ids] = 0
        self.processed = input_ids.shape[1]
        return scores.float() - self.penalty


def _eos_token_ids(model, tokenizer):
    config = model.generation_config
    result = set()
    for ids in (tokenizer.eos_token_id, config.eos_token_id, config.forced_eos_token_id):
        if ids is not None:
            result.update(ids if isinstance(ids, (list, tuple)) else [ids])
    return sorted(result)


def stream_generate(model, tokenizer, prompt, context_length, temperature, top_p, device, skip_special_tokens, stop_on_eos, status_bar=True, ignore_eos=False, penalty_add=0.0, penalty_decay=0.0):
    """Run generate() with TextStreamer. Returns (num_generated, elapsed_seconds, interrupted).
    SIGINT (Ctrl+C) during generation flips a stopping criterion flag, so generation
    halts cleanly after the next token without raising KeyboardInterrupt."""
    penalty_add = _parse_penalty_add(penalty_add)
    penalty_decay = _parse_penalty_decay(penalty_decay)
    generation_kwargs = {}
    if ignore_eos:
        # Keep any existing suppression and cover models with multiple EOS IDs.
        config = model.generation_config
        suppressed = set(config.suppress_tokens or [])
        suppressed.update(_eos_token_ids(model, tokenizer))
        generation_kwargs.update(suppress_tokens=sorted(suppressed), forced_eos_token_id=None)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.size(1)
    if penalty_add > 0:
        generation_kwargs['logits_processor'] = LogitsProcessorList([
            _DecayingRepetitionPenalty(prompt_len, penalty_add, penalty_decay,
                                       _eos_token_ids(model, tokenizer))])

    stopper = _InterruptStoppingCriteria()

    def _on_sigint(signum, frame):
        stopper.interrupted = True

    old_handler = signal.signal(signal.SIGINT, _on_sigint)
    status = _GenerationStatus(enabled=status_bar)
    streamer = _StatusTextStreamer(tokenizer, status, skip_prompt=False, skip_special_tokens=skip_special_tokens)
    try:
        status.refresh(force=True)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_length=context_length,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                streamer=streamer,
                stopping_criteria=StoppingCriteriaList([stopper]),
                eos_token_id=tokenizer.eos_token_id if stop_on_eos and not ignore_eos else None,
                **generation_kwargs,
            )
        elapsed = time.perf_counter() - status.started
        status.tokens = output_ids.size(1) - prompt_len
        status.refresh(force=True)
    finally:
        signal.signal(signal.SIGINT, old_handler)
        status.close()

    num_generated = output_ids.size(1) - prompt_len
    return num_generated, elapsed, stopper.interrupted


def _parse_bool(value):
    if value.lower() in ("true", "1", "yes", "on"):
        return True
    if value.lower() in ("false", "0", "no", "off"):
        return False
    raise ValueError("Use on/off or true/false")


def _parse_penalty_add(value):
    value = float(value)
    if not 0 <= value <= 2:
        raise ValueError("penalty-add must be between 0 and 2")
    return value


def _parse_penalty_decay(value):
    value = float(value)
    if not 0 <= value <= 1:
        raise ValueError("penalty-decay must be between 0 and 1")
    return value


def main():
    parser = argparse.ArgumentParser(description="Stream text generation with RecursiveCompressorLM")
    parser.add_argument("--model-dir", type=str, required=True, help="モデルディレクトリ")
    parser.add_argument("--context-length", type=int, default=1024, help="生成する最大コンテキスト長 (プロンプト含む合計)")
    parser.add_argument("--temperature", type=float, default=1.0, help="サンプリング温度")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p (nucleus) サンプリング閾値 (1.0で無効)")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16", help="推論精度")
    parser.add_argument("--status-bar", action=argparse.BooleanOptionalAction, default=True,
                        help="端末下部に生成トークン数・平均tok/sを表示（既定ON、プロンプト処理時間を含む）")
    parser.add_argument("--ignore-eos", action="store_true",
                        help="EOSトークンを生成候補から除外（対話中にignore-eos on/offで変更可能）")
    parser.add_argument("--penalty-add", type=_parse_penalty_add, default=0.0,
                        help="出力トークンのlogitペナルティ加算量（0～2、既定0で無効）")
    parser.add_argument("--penalty-decay", type=_parse_penalty_decay, default=0.0,
                        help="各生成トークンでのペナルティ保持率（0～1、既定0）")
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
        "skip_special_tokens": True,
        "stop_on_eos": True,
        "ignore_eos": args.ignore_eos,
        "penalty_add": args.penalty_add,
        "penalty_decay": args.penalty_decay,
    }

    print(f"Device: {device}, precision: {args.precision}, context_length: {state['context_length']}, "
          f"temperature: {state['temperature']}, top_p: {state['top_p']}, ignore_eos: {state['ignore_eos']}, "
          f"penalty_add: {state['penalty_add']:g}, penalty_decay: {state['penalty_decay']:g}")
    print("Commands: 'exit', 'temperature [val]', 'top-p [val]', 'context-length [val]', 'skip-special-tokens [true/false]', 'stop-on-eos [true/false]', 'ignore-eos [on/off]', 'penalty-add [0..2]', 'penalty-decay [0..1]'")
    print("Input: Enter to submit, Alt+Enter (or Esc then Enter) for newline")

    commands = {
        "temperature": ("temperature", float),
        "top-p": ("top_p", float),
        "context-length": ("context_length", int),
        "skip-special-tokens": ("skip_special_tokens", _parse_bool),
        "stop-on-eos": ("stop_on_eos", _parse_bool),
        "ignore-eos": ("ignore_eos", _parse_bool),
        "penalty-add": ("penalty_add", _parse_penalty_add),
        "penalty-decay": ("penalty_decay", _parse_penalty_decay),
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

        num_generated, elapsed, interrupted = stream_generate(
            model, tokenizer, prompt,
            state["context_length"], state["temperature"], state["top_p"], device,
            state["skip_special_tokens"],
            state["stop_on_eos"],
            status_bar=args.status_bar,
            ignore_eos=state["ignore_eos"],
            penalty_add=state["penalty_add"],
            penalty_decay=state["penalty_decay"],
        )
        if interrupted:
            print("\n[interrupted]")
        if elapsed > 0 and num_generated > 0:
            print(f"\n[{num_generated} tokens, {elapsed:.2f}s, {num_generated / elapsed:.2f} tok/s]")


if __name__ == "__main__":
    main()
