# Two-layer LogKV + token-stream causal convolution

Protocol: [../../logkv-causal-conv.md](../../logkv-causal-conv.md).
Branch `logkv-causal-conv`, based on main `d707675`. No main merge.

- `common.py`: frozen source, GPU 0/1, task binding and fixed 50k training commands.
- `preflight.py`: shared untrained baseline weights plus new conv parameters, GPU repeat and timing checks.
- `fullsize_smoke.py`: GPU streaming checks, then batch64/T2028 optimizer updates and memory measurement.
- `train_entry.py`: deterministic kernels, task binding, exact initial-state audits.
- `evaluation_smoke.py`: real benchmark checkpoints, independent sample/count verification.
- `run.py`: two independent workers, whole-batch 7.5h cap from GPU preflight, fail-stop, no retries.
- `worker.py`: 50k training, best/final evaluation and unchanged weight hashes.
- `evaluate.py`: standard 41 horizons x256, saved memory/positions/predictions/margins.
- `summarize.py`: CPU independent generator replay and training/metric/configuration audit.
- `compare_baseline.py`: paired per-digit and complete-string rescue/regression against the completed two-layer baseline.

Live root: `/mnt/raid0/RecursiveCompressor/experiments/logkv-causal-conv-20260914/`.
Frozen runtime lives in its `source/`; weights in `exp/<task>/causal-conv4-fixed10-20260914/`.
Status: `campaign.json`, `AGENT_STATUS.json`; task stdout `copying.log`, `selective-copying.log`.
Baseline is read-only `/mnt/raid0/RecursiveCompressor/experiments/logkv-no-position-main-20260913/`.
No additional widths/seeds, per-compression-level convolution, three-layer training or 16M extension are queued.
