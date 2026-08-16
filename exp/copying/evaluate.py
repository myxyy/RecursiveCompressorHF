"""
Evaluate a trained copy-task model across horizons T and plot length
generalization.

Inference uses chunked `step()` with hidden-state carryover (verified
numerically equivalent to a one-shot forward), so memory stays O(batch x
chunk) regardless of T and long horizons can use a decent batch size.

Usage:
    uv run python exp/copying/evaluate.py --run-name base
    uv run python exp/copying/evaluate.py --run-name base --max-t-exp 17 --samples 256

Outputs (next to the checkpoint): results.json, plot.png
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from recursive_compressor_lm import RecursiveCompressorLM  # noqa: E402
from task import MEMORY_LEN, make_batch, score_logits  # noqa: E402

torch.set_float32_matmul_precision("high")

CHUNK_LEN = 8192          # step() feed size (memory bound per call)
TOKEN_BUDGET = 2 ** 19    # ~0.5M tokens per eval batch -> auto batch sizing
                          # (memory per step ~ batch x min(L, CHUNK_LEN);
                          #  2^22 OOMed on 24GB at d_model=512)


def parse_args():
    p = argparse.ArgumentParser(description="Copy task length-generalization eval")
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--samples", type=int, default=256, help="Tごとの評価サンプル数")
    p.add_argument("--max-t-exp", type=int, default=17, help="最大T = 2^この値")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--checkpoint", choices=["auto", "best", "final"], default="auto",
                   help="auto=model_best/があればそれ、なければmodel/ (final)")
    return p.parse_args()


def build_t_grid(max_exp):
    """T=1..14の全点 + 2^k と 1.5*2^k (k=4..max_exp)。"""
    ts = set(range(1, 15))
    for k in range(4, max_exp + 1):
        ts.add(2 ** k)
        if k < max_exp:
            ts.add(3 * 2 ** (k - 1))
    return sorted(ts)


@torch.no_grad()
def eval_horizon(model, T, samples, generator, device, autocast_enabled):
    """Chunked teacher-forced evaluation at horizon T."""
    token_correct = string_correct = token_n = string_n = 0
    L = T + 2 * MEMORY_LEN  # T + 20 (MARKER_LEN - 1 + MEMORY_LEN... = task.seq_len_for)
    batch = max(1, min(samples, TOKEN_BUDGET // max(1, L)))
    done = 0
    while done < samples:
        b = min(batch, samples - done)
        input_ids, _ = make_batch(T, b, generator=generator, device=device)
        hidden = None
        last_logits = None
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=autocast_enabled):
            for i in range(0, input_ids.size(1), CHUNK_LEN):
                logits, hidden = model.step(input_ids[:, i:i + CHUNK_LEN], hidden)
                last_logits = logits
        tok, st, tok_n, st_n = score_logits(last_logits.float(), input_ids)
        token_correct += tok; string_correct += st
        token_n += tok_n; string_n += st_n
        done += b
    return token_correct / token_n, string_correct / string_n


def main():
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        spec = args.device
        device = torch.device(f"cuda:{spec}" if spec.isdigit() else spec)

    data_dir = os.environ.get("DATA_DIR", str(REPO_ROOT / "data"))
    run_dir = Path(data_dir) / "exp" / "copying" / args.run_name
    if args.checkpoint == "best":
        model_dir = run_dir / "model_best"
    elif args.checkpoint == "final":
        model_dir = run_dir / "model"
    else:  # auto
        model_dir = run_dir / "model_best" if (run_dir / "model_best").exists() else run_dir / "model"
    assert model_dir.exists(), f"checkpoint not found: {model_dir}"
    print(f"Using checkpoint: {model_dir.name}")

    run_config = {}
    cfg_path = run_dir / "run_config.json"
    if cfg_path.exists():
        run_config = json.loads(cfg_path.read_text())
    train_max_t = run_config.get("max_t")

    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    model = RecursiveCompressorLM.from_pretrained(model_dir).to(device=device, dtype=dtype)
    model.eval()
    autocast_enabled = device.type == "cuda" and args.precision == "bf16"

    t_grid = build_t_grid(args.max_t_exp)
    print(f"Evaluating {model_dir} on {len(t_grid)} horizons "
          f"(T=1..{t_grid[-1]}), {args.samples} samples each, device={device}")

    generator = torch.Generator().manual_seed(args.seed)
    results = {}
    for T in t_grid:
        t0 = time.time()
        tok_acc, str_acc = eval_horizon(model, T, args.samples, generator, device,
                                        autocast_enabled)
        results[T] = {"token_acc": tok_acc, "string_acc": str_acc, "n": args.samples}
        marker = " <= train horizon" if train_max_t and T <= train_max_t else ""
        print(f"T={T:>7} | token {tok_acc:.4f} | string {str_acc:.4f} | "
              f"{time.time()-t0:.1f}s{marker}", flush=True)

    out = {"train_max_t": train_max_t, "samples": args.samples,
           "precision": args.precision, "results": results}
    results_path = run_dir / "results.json"
    results_path.write_text(json.dumps(out, indent=2))
    print(f"Saved {results_path}")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = list(results.keys())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ts, [results[t]["token_acc"] for t in ts], "o-", label="token accuracy", ms=4)
    ax.plot(ts, [results[t]["string_acc"] for t in ts], "s-", label="string accuracy", ms=4)
    if train_max_t:
        ax.axvline(train_max_t, color="gray", ls="--", lw=1,
                   label=f"train horizon (T={train_max_t})")
    ax.axhline(1 / 8, color="lightgray", ls=":", lw=1, label="chance (1/8)")
    ax.set_xscale("log")
    ax.set_xlabel("T (memory horizon, log scale)")
    ax.set_ylabel("accuracy on last 10 tokens")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Copy task length generalization — {args.run_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    plot_path = run_dir / "plot.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
