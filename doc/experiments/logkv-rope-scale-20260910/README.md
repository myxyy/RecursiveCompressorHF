# RoPE angle-scale Copying comparison

Protocol: [../../logkv-rope-scale.md](../../logkv-rope-scale.md).

- `common.py`: paths, scales, old training command and fixed evaluation horizons.
- `preflight.py` / `preflight.json`: matched initialization, RNG, legacy compatibility and GPU smoke.
- `evaluation_smoke.json`: standard/per-digit/paired evaluator end-to-end smoke on an untrained model.
- `run.py`: 3 independent GPU workers, total 7.5h deadline, fail-stop process groups, no retries.
- `worker.py`: unchanged 50k training protocol, then best/final evaluation.
- `evaluate.py`: standard 41×256 and paired 34×32 evaluation per checkpoint, per-digit outputs.
- `summarize.py`: CPU count/config/weight audit; compact archive and comparison plot.

Live status: `/mnt/raid0/RecursiveCompressor/experiments/logkv-rope-scale-20260910/campaign.json`.
Per-mode status/logs: `<root>/alpha-{1,2over3,half}/`.
Weights and training logs: `<root>/exp/copying/<mode>-fixed10-20260910/`.
Stop the supervisor with SIGTERM to stop its workers and child jobs.
No Selective, extra seeds, 16M evaluation or main merge is queued.

Completed and reviewed 2026-09-11. All3 runs/450 cells passed audit; GPUs released.
Results and interpretation: [../../logkv-rope-scale.md](../../logkv-rope-scale.md).
`review.py` independently rechecks archived counts/hashes and full-size legacy alpha1 CPU compatibility;
output: `results/manual_review.json`. `campaign.json` preserves the original completion record;
RAID `AGENT_STATUS.json` records the subsequent reviewed status.
