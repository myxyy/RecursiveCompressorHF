"""
Train RecursiveCompressorLM on the Copy Memory Problem.

Single-GPU, online sample generation (a fresh batch every step; no dataset,
no epochs). T is sampled uniformly per step (fixed within a batch, so no
padding is needed).

Usage:
    uv run python exp/copying/train.py --run-name base
    uv run python exp/copying/train.py --run-name small --max-t 44 --steps 3000

Outputs (under $DATA_DIR/exp/copying/{run_name}/):
    model/            save_pretrained checkpoint (+ run_config.json)
    train_log.jsonl   per-log-interval loss/accuracy records
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from configuration_recursive_compressor import RecursiveCompressorConfig  # noqa: E402
from recursive_compressor_lm import RecursiveCompressorLM  # noqa: E402
from task import VOCAB_SIZE, make_batch, mask_non_answer, score_logits  # noqa: E402

torch.set_float32_matmul_precision("high")


def parse_args():
    p = argparse.ArgumentParser(description="Copy task training")
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--max-t", type=int, default=2028,
                   help="訓練時のTの上限 (T ~ U[1, max_t]、系列長は T+20)")
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=1000, help="線形warmupステップ数")
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--d-ff", type=int, default=1024)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--chunk-size", type=int, default=4)
    p.add_argument("--compress-size", type=int, default=1)
    p.add_argument("--retrieve-size", type=int, default=4)
    p.add_argument("--loss-positions", choices=["all", "answer"], default="all",
                   help="all=全位置CE (CKConv等の既存研究と同じ) / answer=末尾10位置のみ")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None, help="例: 0, cuda:3, cpu")
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--eval-interval", type=int, default=5000,
                   help="訓練中の簡易汎化チェック間隔 (0で無効)")
    p.add_argument("--save-interval", type=int, default=10000)
    return p.parse_args()


def resolve_device(spec):
    if spec is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec.isdigit():
        spec = f"cuda:{spec}"
    return torch.device(spec)


@torch.no_grad()
def quick_eval(model, device, ts, samples, generator, autocast_dtype):
    """簡易評価: 各Tでstring accuracyを返す (訓練中の進捗確認用)。"""
    model.eval()
    out = {}
    for T in ts:
        input_ids, _ = make_batch(T, samples, generator=generator, device=device)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                            enabled=device.type == "cuda"):
            logits = model(input_ids).logits
        _, string_correct, _, n = score_logits(logits.float(), input_ids)
        out[T] = string_correct / n
    model.train()
    return out


def main():
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")
    device = resolve_device(args.device)

    data_dir = os.environ.get("DATA_DIR", str(REPO_ROOT / "data"))
    run_dir = Path(data_dir) / "exp" / "copying" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    config = RecursiveCompressorConfig(
        vocab_size=VOCAB_SIZE,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        chunk_size=args.chunk_size,
        compress_size=args.compress_size,
        retrieve_size=args.retrieve_size,
        num_layers=args.num_layers,
        pad_token_id=None, bos_token_id=None, eos_token_id=None,
    )
    model = RecursiveCompressorLM(config).to(device)
    model.train()
    num_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    autocast_dtype = torch.bfloat16

    run_config = {**vars(args), "num_params": num_params, "loss": "position-aligned CE"}
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"Copy task training: {num_params:,} params on {device}, "
          f"T~U[1,{args.max_t}], {args.steps} steps, loss={args.loss_positions}")
    print(f"Run dir: {run_dir}")

    data_gen = torch.Generator().manual_seed(args.seed + 1)
    eval_gen = torch.Generator().manual_seed(args.seed + 2)
    eval_ts = sorted({max(1, args.max_t // 4), args.max_t // 2, args.max_t,
                      args.max_t * 2, args.max_t * 4})

    log_path = run_dir / "train_log.jsonl"
    log_f = open(log_path, "a")
    ema_loss = None
    interval_tok = interval_tok_n = interval_str = interval_str_n = 0
    t0 = time.time()

    for step in range(1, args.steps + 1):
        T = int(torch.randint(1, args.max_t + 1, (1,), generator=data_gen).item())
        input_ids, labels = make_batch(T, args.batch_size, generator=data_gen, device=device)
        if args.loss_positions == "answer":
            labels = mask_non_answer(labels)

        lr = args.lr * min(1.0, step / max(1, args.warmup))
        for g in optimizer.param_groups:
            g["lr"] = lr

        with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                            enabled=device.type == "cuda"):
            out = model(input_ids, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        loss = out.loss.item()
        ema_loss = loss if ema_loss is None else 0.99 * ema_loss + 0.01 * loss
        tok, st, tok_n, st_n = score_logits(out.logits.float(), input_ids)
        interval_tok += tok; interval_tok_n += tok_n
        interval_str += st; interval_str_n += st_n

        if step % args.log_interval == 0:
            tok_acc = interval_tok / max(1, interval_tok_n)
            str_acc = interval_str / max(1, interval_str_n)
            elapsed = time.time() - t0
            rec = {"step": step, "loss": loss, "ema_loss": ema_loss,
                   "token_acc": tok_acc, "string_acc": str_acc, "lr": lr,
                   "elapsed_sec": round(elapsed, 1)}
            log_f.write(json.dumps(rec) + "\n"); log_f.flush()
            print(f"step {step}/{args.steps} | loss {ema_loss:.4f} | "
                  f"tok_acc {tok_acc:.4f} | str_acc {str_acc:.4f} | "
                  f"lr {lr:.2e} | {elapsed:.0f}s", flush=True)
            interval_tok = interval_tok_n = interval_str = interval_str_n = 0

        if args.eval_interval and step % args.eval_interval == 0:
            accs = quick_eval(model, device, eval_ts, 64, eval_gen, autocast_dtype)
            msg = " ".join(f"T={t}:{a:.3f}" for t, a in accs.items())
            log_f.write(json.dumps({"step": step, "quick_eval": accs}) + "\n"); log_f.flush()
            print(f"  quick_eval (string acc): {msg}", flush=True)

        if step % args.save_interval == 0 or step == args.steps:
            model.save_pretrained(run_dir / "model")

    log_f.close()
    print(f"Done. Model saved to {run_dir / 'model'}")


if __name__ == "__main__":
    main()
