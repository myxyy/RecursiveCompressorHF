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

from configuration_logkv import LogKVConfig  # noqa: E402
from configuration_recursive_compressor import RecursiveCompressorConfig  # noqa: E402
from logkv_lm import LogKVLM  # noqa: E402
from recursive_compressor_lm import RecursiveCompressorLM  # noqa: E402
from task import TASK_NAME, VOCAB_SIZE, make_batch, mask_non_answer, score_logits  # noqa: E402

torch.set_float32_matmul_precision("high")


def parse_args():
    p = argparse.ArgumentParser(description="Copy task training")
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--arch", choices=["recursive", "logkv"], default="recursive",
                   help="recursive=RecursiveCompressorLM / logkv=LogKVLM "
                        "(compress-size/retrieve-sizeはlogkvでは無視)")
    p.add_argument("--phase-emb", action="store_true",
                   help="logkv: 学習可能位相埋め込み(位置のC進数桁)を有効化")
    p.add_argument("--phase-levels", type=int, default=16,
                   help="logkv: 位相埋め込みに使う桁数(周期 C^phase_levels)。訓練で全桁値が"
                        "出現する範囲に制限すると域外でも未学習ベクトルを踏まない")
    p.add_argument("--learnable-decay", action="store_true",
                   help="レベル減衰の係数(初期値log C)をヘッド・層ごとに学習可能にする")
    p.add_argument("--max-t", type=int, default=2028,
                   help="訓練時のTの上限 (T ~ U[1, max_t]、系列長は T+20)")
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=64,
                   help="実効バッチサイズ (grad-accum使用時はmicro batch = batch/accum)")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="勾配蓄積数。バッチをこの数のマイクロバッチに分けてメモリを削減"
                        " (同一Tで蓄積するので実効バッチの意味は不変)")
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
    p.add_argument("--t-dist", choices=["uniform", "loguniform"], default="uniform",
                   help="訓練Tのサンプリング分布。loguniformは短いTを多く出す"
                        " (T~U[1,2028]一様ではanswer信号が薄く短T足場も出ず学習が立ち上がらない)")
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
    """簡易評価: 各Tでstring accuracyを返す (訓練中の進捗確認用)。
    チャンク分割step推論 (evaluate.pyと同方式)。大きいTの一発forwardは
    メモリ・CUDAカーネル形状制限 (invalid configuration) を踏むため。"""
    model.eval()
    CHUNK_LEN = 4096
    TOKEN_BUDGET = 2 ** 18
    out = {}
    for T in ts:
        L = T + 20
        bs = max(1, min(samples, TOKEN_BUDGET // max(1, L)))
        string_correct = n = 0
        done = 0
        while done < samples:
            b = min(bs, samples - done)
            input_ids, labels = make_batch(T, b, generator=generator, device=device)
            hidden = None
            last_logits = None
            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                enabled=device.type == "cuda"):
                for i in range(0, input_ids.size(1), CHUNK_LEN):
                    logits, hidden = model.step(input_ids[:, i:i + CHUNK_LEN], hidden)
                    last_logits = logits
            _, sc, _, sn = score_logits(last_logits.float(), labels)
            string_correct += sc; n += sn
            done += b
        out[T] = string_correct / n
    model.train()
    return out


def main():
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")
    device = resolve_device(args.device)

    data_dir = os.environ.get("DATA_DIR", str(REPO_ROOT / "data"))
    run_dir = Path(data_dir) / "exp" / TASK_NAME / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    if args.arch == "logkv":
        config = LogKVConfig(
            vocab_size=VOCAB_SIZE,
            d_model=args.d_model,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            chunk_size=args.chunk_size,
            num_layers=args.num_layers,
            phase_emb=args.phase_emb,
            phase_levels=args.phase_levels,
            learnable_decay=args.learnable_decay,
            pad_token_id=None, bos_token_id=None, eos_token_id=None,
        )
        model = LogKVLM(config).to(device)
    else:
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

    print(f"Copy task training ({args.arch}): {num_params:,} params on {device}, "
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
    # Best-checkpoint tracking: training occasionally destabilizes for a few
    # hundred steps (loss spikes, accuracy craters, then recovers), so the
    # final checkpoint can land in a dip. Keep the best interval-averaged
    # weights in model_best/ (lexicographic: string acc, token acc, -EMA loss).
    best_score = (-1.0, -1.0, float("-inf"))
    t0 = time.time()

    assert args.batch_size % args.grad_accum == 0, "batch_size must be divisible by grad_accum"
    micro_bs = args.batch_size // args.grad_accum

    for step in range(1, args.steps + 1):
        if args.t_dist == "loguniform":
            u = torch.rand(1, generator=data_gen).item()
            T = max(1, min(args.max_t, int(math.exp(u * math.log(args.max_t + 1)))))
        else:
            T = int(torch.randint(1, args.max_t + 1, (1,), generator=data_gen).item())

        lr = args.lr * min(1.0, step / max(1, args.warmup))
        for g in optimizer.param_groups:
            g["lr"] = lr

        # Gradient accumulation over micro-batches that share the same T, so
        # the effective batch is identical to a single batch_size-sized batch.
        loss = 0.0
        for _ in range(args.grad_accum):
            input_ids, labels = make_batch(T, micro_bs, generator=data_gen, device=device)
            if args.loss_positions == "answer":
                labels = mask_non_answer(labels)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                enabled=device.type == "cuda"):
                out = model(input_ids, labels=labels)
            (out.loss / args.grad_accum).backward()
            loss += out.loss.item() / args.grad_accum
            tok, st, tok_n, st_n = score_logits(out.logits.float(), labels)
            interval_tok += tok; interval_tok_n += tok_n
            interval_str += st; interval_str_n += st_n
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        ema_loss = loss if ema_loss is None else 0.99 * ema_loss + 0.01 * loss

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

            score = (str_acc, tok_acc, -ema_loss)
            if score > best_score:
                best_score = score
                model.save_pretrained(run_dir / "model_best")
                with open(run_dir / "best.json", "w") as f:
                    json.dump({"step": step, "string_acc": str_acc,
                               "token_acc": tok_acc, "ema_loss": ema_loss}, f, indent=2)

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
