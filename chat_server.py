"""
Chat WebUI server for RecursiveCompressorLM (instruct).

Loads the instruct model (default: myxy/recursive-compressor-v1.2-2.6b-c4c1-instruct)
and serves a ChatGPT-like interface in the browser. Each browser tab opens
its own WebSocket; the per-tab hidden state lives only as long as that
WebSocket is open. Inference runs sequentially across all sessions
(one global lock).

Usage:
    uv run python chat_server.py
    uv run python chat_server.py --model-dir /path/to/local/checkpoint
    uv run python chat_server.py --port 8080 --host 0.0.0.0
"""

import argparse
import asyncio
import threading
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from predict import _DTYPES, _load_model, _load_tokenizer

DEFAULT_MODEL = "myxy/recursive-compressor-v1.2-2.6b-c4c1-instruct"
DEFAULT_PORT = 8000
MAX_NEW_TOKENS_PER_TURN = 1024

torch.set_float32_matmul_precision("high")


class ServerState:
    model = None
    tokenizer = None
    device = None
    bos_id = None
    eos_id = None
    inference_lock = threading.Lock()
    # Default slider values served to the browser (overridable via CLI).
    default_temperature = 1.0
    default_top_p = 1.0
    default_penalty_add = 0.0
    default_penalty_decay = 0.0


state = ServerState()


@torch.no_grad()
def _sample_token(logits: torch.Tensor, temperature: float, top_p: float,
                  penalty: torch.Tensor = None) -> int:
    """Sample one token from a 1-D logits tensor.
    penalty: optional (vocab,) tensor subtracted from logits before the
    temperature division (decaying repetition penalty)."""
    logits = logits.float()
    if penalty is not None:
        logits = logits - penalty
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
    if 0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep = cumulative <= top_p
        keep[..., 0] = True
        mask = torch.zeros_like(probs)
        mask[sorted_idx[keep]] = 1.0
        probs = probs * mask
        s = probs.sum()
        if s <= 0:
            return int(sorted_idx[0].item())
        probs = probs / s
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def _generate_turn(hidden, query_text, temperature, top_p, on_token, cancel_event=None,
                   penalty=None, penalty_add=0.0, penalty_decay=0.0):
    """Run one turn in Llama 2 format. Each turn is wrapped as
    `<s>[INST]q[/INST]a</s>` so the input here is `<s>[INST]q[/INST]` and
    generation stops at EOS. The EOS is fed through `step` before stopping so
    the next turn's input cleanly begins with `<s>[INST]...`.

    cancel_event: optional threading.Event; when set, generation stops early
    (checked before the prompt step and each token). The returned hidden is
    then mid-turn and meant to be discarded by the caller (reset/disconnect).

    penalty: (vocab,) fp32 tensor of accumulated repetition penalties, or None.
    Decaying repetition penalty: sampling uses softmax((logits - penalty)/T);
    after each generated token w, `penalty *= penalty_decay` (all entries, so
    unused words recover over time) then `penalty[w] += penalty_add`. EOS is
    exempt (never penalized) so accumulated penalty cannot suppress stopping.
    Only generated tokens update the penalty — prompt tokens do not.
    The tensor is mutated in place; the caller keeps/discards it alongside
    the hidden state.

    Returns (new_hidden, reason, num_generated) where reason is one of
    "eos" (model emitted EOS), "max_tokens" (forced cutoff), or
    "cancelled" (cancel_event was set)."""
    def cancelled():
        return cancel_event is not None and cancel_event.is_set()

    use_penalty = penalty is not None and penalty_add > 0

    text = f"[INST]{query_text}[/INST]"
    ids = [state.bos_id] + state.tokenizer.encode(text, add_special_tokens=False)

    if cancelled():
        return hidden, "cancelled", 0

    input_ids = torch.tensor([ids], dtype=torch.long, device=state.device)
    logits, hidden = state.model.step(input_ids, hidden)
    next_logits = logits[0, -1, :]

    num_generated = 0
    reason = "max_tokens"
    for _ in range(MAX_NEW_TOKENS_PER_TURN):
        if cancelled():
            reason = "cancelled"
            break
        next_id = _sample_token(next_logits, temperature, top_p,
                                penalty if use_penalty else None)
        if next_id == state.eos_id:
            _, hidden = state.model.step(
                torch.tensor([[next_id]], dtype=torch.long, device=state.device), hidden
            )
            reason = "eos"
            break
        if use_penalty:
            penalty.mul_(penalty_decay)
            penalty[next_id] += penalty_add
        on_token(next_id)
        num_generated += 1
        logits, hidden = state.model.step(
            torch.tensor([[next_id]], dtype=torch.long, device=state.device), hidden
        )
        next_logits = logits[0, -1, :]
    else:
        # Hit the per-turn cap without emitting EOS. Inject one so the running
        # token stream stays in `<s>[INST]q[/INST]a</s><s>...` shape.
        _, hidden = state.model.step(
            torch.tensor([[state.eos_id]], dtype=torch.long, device=state.device), hidden
        )

    return hidden, reason, num_generated


HTML_PAGE = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RecursiveCompressor Chat</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", "Hiragino Sans", sans-serif;
         margin: 0; background: #f6f6f7; color: #222; height: 100vh; display: flex; flex-direction: column; }
  header { padding: 12px 16px; background: #fff; border-bottom: 1px solid #ddd; }
  header h1 { margin: 0 0 8px; font-size: 18px; }
  .controls { display: flex; gap: 24px; flex-wrap: wrap; align-items: center; }
  .slider { display: flex; align-items: center; gap: 8px; font-size: 13px; }
  .slider input[type=range] { width: 180px; }
  .slider .val { display: inline-block; min-width: 36px; text-align: right; font-variant-numeric: tabular-nums; }
  #status { font-size: 12px; color: #888; margin-left: auto; }
  #chat { flex: 1; overflow-y: auto; padding: 16px; max-width: 900px; width: 100%;
          margin: 0 auto; }
  .msg { margin: 12px 0; padding: 10px 14px; border-radius: 12px; max-width: 80%;
         white-space: pre-wrap; word-wrap: break-word; line-height: 1.5; }
  .user { background: #d0e9ff; margin-left: auto; }
  .bot  { background: #fff; border: 1px solid #e2e2e2; }
  .meta { font-size: 11px; color: #999; margin-top: 6px; }
  footer { padding: 12px; background: #fff; border-top: 1px solid #ddd; }
  .input-row { display: flex; gap: 8px; max-width: 900px; margin: 0 auto; }
  textarea { flex: 1; padding: 8px 10px; font-size: 14px; border: 1px solid #ccc;
             border-radius: 8px; resize: none; font-family: inherit; min-height: 44px; max-height: 160px; }
  button { padding: 8px 18px; font-size: 14px; border: none; border-radius: 8px;
           background: #2b6cb0; color: #fff; cursor: pointer; }
  button:disabled { background: #aaa; cursor: not-allowed; }
  button.secondary { background: #e2e2e2; color: #333; }
  button.secondary:hover:not(:disabled) { background: #d4d4d4; }
  button.secondary:disabled { background: #eee; color: #aaa; }
  details.penalty { position: relative; font-size: 13px; }
  details.penalty summary { cursor: pointer; user-select: none; list-style: none;
                            padding: 4px 10px; border: 1px solid #ccc; border-radius: 8px;
                            background: #fafafa; }
  details.penalty summary::-webkit-details-marker { display: none; }
  details.penalty summary::after { content: " ▾"; color: #888; }
  details.penalty[open] summary { background: #eef4fb; border-color: #2b6cb0; }
  .penalty-panel { position: absolute; top: calc(100% + 6px); left: 0; z-index: 10;
                   background: #fff; border: 1px solid #ccc; border-radius: 8px;
                   padding: 12px 14px; box-shadow: 0 4px 12px rgba(0,0,0,0.12);
                   display: flex; flex-direction: column; gap: 10px; white-space: nowrap; }
  .gh-link { margin-left: 10px; color: #555; display: inline-flex; vertical-align: middle; }
  .gh-link:hover { color: #000; }
  .gh-link svg { width: 22px; height: 22px; fill: currentColor; }
</style>
</head>
<body>
<header>
  <h1>RecursiveCompressor Chat<a class="gh-link" href="https://github.com/myxyy/RecursiveCompressorHF"
      target="_blank" rel="noopener" title="GitHub: myxyy/RecursiveCompressorHF"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg></a></h1>
  <div class="controls">
    <label class="slider">Temperature
      <input type="range" id="temp" min="0" max="2" step="0.01" value="__TEMP_DEFAULT__">
      <span class="val" id="temp-val">__TEMP_DEFAULT__</span>
    </label>
    <label class="slider">Top-p
      <input type="range" id="topp" min="0" max="1" step="0.01" value="__TOPP_DEFAULT__">
      <span class="val" id="topp-val">__TOPP_DEFAULT__</span>
    </label>
    <details class="penalty">
      <summary>Repeat penalty: <span id="pen-summary"></span></summary>
      <div class="penalty-panel">
        <label class="slider" title="出力トークンのlogitから引くペナルティ加算量 (0で無効)">Add
          <input type="range" id="pen-add" min="0" max="2" step="0.01" value="__PEN_ADD_DEFAULT__">
          <span class="val" id="pen-add-val">__PEN_ADD_DEFAULT__</span>
        </label>
        <label class="slider" title="毎トークンごとの全ペナルティ減衰係数 (0で直前トークンのみ、1で無減衰蓄積)">Decay
          <input type="range" id="pen-decay" min="0" max="1" step="0.01" value="__PEN_DECAY_DEFAULT__">
          <span class="val" id="pen-decay-val">__PEN_DECAY_DEFAULT__</span>
        </label>
      </div>
    </details>
    <span id="status">connecting...</span>
  </div>
</header>
<div id="chat"></div>
<footer>
  <div class="input-row">
    <textarea id="input" placeholder="メッセージを入力 (Enter で送信、Shift+Enter で改行)"></textarea>
    <button id="reset" class="secondary" disabled title="会話をリセット (進行中の生成は中断、スライダー設定は維持、蓄積ペナルティはクリア)">リセット</button>
    <button id="send" disabled>送信</button>
  </div>
</footer>
<script>
const tempEl = document.getElementById('temp');
const tempVal = document.getElementById('temp-val');
const toppEl = document.getElementById('topp');
const toppVal = document.getElementById('topp-val');
const statusEl = document.getElementById('status');
const chatEl = document.getElementById('chat');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send');
const resetBtn = document.getElementById('reset');
const penAddEl = document.getElementById('pen-add');
const penAddVal = document.getElementById('pen-add-val');
const penDecayEl = document.getElementById('pen-decay');
const penDecayVal = document.getElementById('pen-decay-val');
const penSummary = document.getElementById('pen-summary');

tempEl.addEventListener('input', () => { tempVal.textContent = (+tempEl.value).toFixed(2); });
toppEl.addEventListener('input', () => { toppVal.textContent = (+toppEl.value).toFixed(2); });

function updatePenaltyLabels() {
  penAddVal.textContent = (+penAddEl.value).toFixed(2);
  penDecayVal.textContent = (+penDecayEl.value).toFixed(2);
  penSummary.textContent = +penAddEl.value > 0
    ? (+penAddEl.value).toFixed(2) + ' / ' + (+penDecayEl.value).toFixed(2)
    : 'off';
}
penAddEl.addEventListener('input', updatePenaltyLabels);
penDecayEl.addEventListener('input', updatePenaltyLabels);
updatePenaltyLabels();

let ws = null;
let busy = false;
let currentBot = null;
let awaitingReset = false;

function setStatus(text, color) {
  statusEl.textContent = text;
  statusEl.style.color = color || '#888';
}

function updateButtons() {
  const open = ws && ws.readyState === WebSocket.OPEN;
  // Send is blocked while generating, resetting, or disconnected.
  sendBtn.disabled = busy || awaitingReset || !open;
  // Reset stays enabled during generation so it can interrupt it.
  resetBtn.disabled = awaitingReset || !open;
}

function setBusy(b) {
  busy = b;
  updateButtons();
}

function addMessage(role, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = text;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  return div;
}

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(proto + '://' + location.host + '/ws/chat');
  ws.onopen = () => { awaitingReset = false; setStatus('connected', '#0a0'); setBusy(false); };
  ws.onclose = () => { setStatus('disconnected (reload to reconnect)', '#c00'); setBusy(true); };
  ws.onerror = () => { setStatus('connection error', '#c00'); };
  ws.onmessage = (ev) => {
    const data = JSON.parse(ev.data);
    if (data.type === 'reset_done') {
      awaitingReset = false;
      setStatus('connected', '#0a0');
      setBusy(false);
      return;
    }
    // While a reset is pending, ignore any in-flight generation messages
    // (start/delta/end) from the turn being interrupted.
    if (awaitingReset) return;
    if (data.type === 'start') {
      currentBot = addMessage('bot', '');
      setStatus('generating...', '#06a');
    } else if (data.type === 'delta') {
      if (currentBot) {
        currentBot.textContent += data.text;
        chatEl.scrollTop = chatEl.scrollHeight;
      }
    } else if (data.type === 'end') {
      if (currentBot) {
        const meta = document.createElement('div');
        meta.className = 'meta';
        meta.textContent = data.num_tokens + ' tokens (' + data.reason + ')';
        currentBot.appendChild(meta);
        currentBot = null;
      }
      setBusy(false);
      setStatus('connected', '#0a0');
    } else if (data.type === 'error') {
      addMessage('bot', '[error] ' + data.message);
      setBusy(false);
      setStatus('error', '#c00');
    }
  };
}

function send() {
  const text = inputEl.value.trim();
  if (!text || busy || !ws || ws.readyState !== WebSocket.OPEN) return;
  addMessage('user', text);
  inputEl.value = '';
  ws.send(JSON.stringify({
    type: 'query',
    text: text,
    temperature: +tempEl.value,
    top_p: +toppEl.value,
    penalty_add: +penAddEl.value,
    penalty_decay: +penDecayEl.value,
  }));
  setBusy(true);
  setStatus('queued / generating...', '#06a');
}

function doReset() {
  if (!ws || ws.readyState !== WebSocket.OPEN || awaitingReset) return;
  // Interrupt any in-progress generation and clear the conversation.
  // temperature / top-p sliders are left untouched, so they carry over.
  ws.send(JSON.stringify({ type: 'reset' }));
  chatEl.innerHTML = '';
  currentBot = null;
  awaitingReset = true;
  setStatus('resetting...', '#06a');
  updateButtons();
}

sendBtn.addEventListener('click', send);
resetBtn.addEventListener('click', doReset);
inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});

// Closing/refreshing the tab closes the socket promptly so the server can
// interrupt the in-progress generation (also handled by the browser on unload).
window.addEventListener('pagehide', () => { if (ws) ws.close(); });

connect();
</script>
</body>
</html>
"""


app = FastAPI()


@app.get("/")
def index():
    html = (
        HTML_PAGE
        .replace("__TEMP_DEFAULT__", f"{state.default_temperature:g}")
        .replace("__TOPP_DEFAULT__", f"{state.default_top_p:g}")
        .replace("__PEN_ADD_DEFAULT__", f"{state.default_penalty_add:g}")
        .replace("__PEN_DECAY_DEFAULT__", f"{state.default_penalty_decay:g}")
    )
    return HTMLResponse(html)


@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()

    # Per-connection state. `hidden` is the conversation context (None = fresh).
    # `penalty` is the (vocab,) decaying-repetition-penalty tensor (None until
    # first used); it lives and dies with the conversation, like `hidden`.
    # `cancel_event` interrupts the in-progress generation (reset / disconnect).
    session = {"hidden": None, "penalty": None}
    cancel_event = threading.Event()
    gen_task = None  # asyncio.Task streaming the current generation, or None

    async def run_generation(text, temperature, top_p, penalty_add, penalty_decay):
        """Stream one turn to the client. Runs the model in a worker thread and
        forwards tokens via an asyncio.Queue. Saves the new hidden state only if
        the turn was not cancelled."""
        cancel_event.clear()
        await ws.send_json({"type": "start"})

        # Work on a copy of the penalty tensor so a cancelled turn rolls back
        # penalties together with the hidden state.
        if penalty_add > 0:
            if session["penalty"] is not None:
                penalty = session["penalty"].clone()
            else:
                penalty = torch.zeros(
                    state.model.config.vocab_size, dtype=torch.float32, device=state.device
                )
        else:
            penalty = None  # disabled this turn; keep the session tensor as-is

        q: asyncio.Queue = asyncio.Queue()

        def post(item):
            # Hand an item from the worker thread to the asyncio queue. The
            # worker can outlive the connection; if the loop is gone, drop the
            # item and close the orphaned coroutine to avoid a warning.
            coro = q.put(item)
            try:
                asyncio.run_coroutine_threadsafe(coro, loop)
            except RuntimeError:
                coro.close()

        def on_token_cb(tid: int):
            post(("token", tid))

        def runner(prev_hidden):
            with state.inference_lock:
                try:
                    result = _generate_turn(
                        prev_hidden, text, temperature, top_p, on_token_cb, cancel_event,
                        penalty=penalty, penalty_add=penalty_add, penalty_decay=penalty_decay,
                    )
                except Exception as e:
                    post(("error", repr(e)))
                    return
            post(("done", *result))

        threading.Thread(target=runner, args=(session["hidden"],), daemon=True).start()

        collected_ids: list[int] = []
        decoded_so_far = ""
        while True:
            event = await q.get()
            kind = event[0]
            if kind == "token":
                collected_ids.append(event[1])
                if cancel_event.is_set():
                    continue  # drain remaining tokens silently; turn is being discarded
                full = state.tokenizer.decode(collected_ids, skip_special_tokens=True)
                delta = full[len(decoded_so_far):]
                if delta:
                    decoded_so_far = full
                    await ws.send_json({"type": "delta", "text": delta})
            elif kind == "done":
                _, new_hidden, reason, num_gen = event
                if reason != "cancelled":
                    session["hidden"] = new_hidden
                    if penalty is not None:
                        session["penalty"] = penalty
                    await ws.send_json({"type": "end", "reason": reason, "num_tokens": num_gen})
                break
            elif kind == "error":
                await ws.send_json({"type": "error", "message": event[1]})
                break

    try:
        while True:
            msg = await ws.receive_json()
            t = msg.get("type")

            if t == "reset":
                # Interrupt any in-progress generation, then drop the context.
                cancel_event.set()
                if gen_task is not None:
                    try:
                        await gen_task
                    except Exception:
                        pass
                    gen_task = None
                session["hidden"] = None
                session["penalty"] = None
                await ws.send_json({"type": "reset_done"})

            elif t == "query":
                if gen_task is not None and not gen_task.done():
                    continue  # already generating (client also guards via busy)
                text = (msg.get("text") or "").strip()
                if not text:
                    continue
                try:
                    temperature = float(msg.get("temperature", 1.0))
                    top_p = float(msg.get("top_p", 1.0))
                    penalty_add = float(msg.get("penalty_add", 0.0))
                    penalty_decay = float(msg.get("penalty_decay", 0.0))
                except (TypeError, ValueError):
                    temperature, top_p = 1.0, 1.0
                    penalty_add, penalty_decay = 0.0, 0.0
                temperature = max(0.0, min(2.0, temperature))
                top_p = max(0.0, min(1.0, top_p))
                penalty_add = max(0.0, min(2.0, penalty_add))
                penalty_decay = max(0.0, min(1.0, penalty_decay))
                gen_task = asyncio.create_task(
                    run_generation(text, temperature, top_p, penalty_add, penalty_decay)
                )

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        # Interrupt the worker thread and drop the context. Hidden state goes
        # out of scope; the CUDA caching allocator reclaims it.
        cancel_event.set()
        if gen_task is not None:
            gen_task.cancel()
        session["hidden"] = None
        session["penalty"] = None


def main():
    parser = argparse.ArgumentParser(description="Chat WebUI server for RecursiveCompressorLM")
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL,
                        help=f"モデルパス (HFリポジトリID or ローカルディレクトリ)。デフォルト: {DEFAULT_MODEL}")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"ポート (デフォルト {DEFAULT_PORT})")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="バインドホスト (デフォルト 127.0.0.1)")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16", help="推論精度")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="温度スライダーの初期値 (デフォルト 1.0)")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="top-p スライダーの初期値 (デフォルト 1.0)")
    parser.add_argument("--penalty-add", type=float, default=0.0,
                        help="繰り返しペナルティ加算量の初期値 (0で無効、デフォルト 0)")
    parser.add_argument("--penalty-decay", type=float, default=0.0,
                        help="繰り返しペナルティ減衰係数の初期値 (0~1、デフォルト 0)")
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

    print(f"Loading model from {args.model_dir} ...", flush=True)
    model = _load_model(args.model_dir, device, dtype=_DTYPES[args.precision])
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    tokenizer = _load_tokenizer(args.model_dir)
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    if bos_id is None or eos_id is None:
        raise RuntimeError("Tokenizer needs both bos_token_id and eos_token_id for Llama2-style turn framing.")

    state.model = model
    state.tokenizer = tokenizer
    state.device = device
    state.bos_id = bos_id
    state.eos_id = eos_id
    # Clamp to the slider ranges so the served initial value is always valid.
    state.default_temperature = max(0.0, min(2.0, args.temperature))
    state.default_top_p = max(0.0, min(1.0, args.top_p))
    state.default_penalty_add = max(0.0, min(2.0, args.penalty_add))
    state.default_penalty_decay = max(0.0, min(1.0, args.penalty_decay))

    print(f"Device: {device}, precision: {args.precision}, BOS id: {bos_id}, EOS id: {eos_id}")
    print(f"Default sliders: temperature={state.default_temperature:g}, top_p={state.default_top_p:g}, "
          f"penalty_add={state.default_penalty_add:g}, penalty_decay={state.default_penalty_decay:g}")
    print(f"Listening on http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
