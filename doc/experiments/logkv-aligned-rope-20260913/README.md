# End-anchored RoPE Copying experiment

Protocol: [../../logkv-aligned-rope.md](../../logkv-aligned-rope.md).

- `common.py`: fixed paths, paired horizons and matched training commands.
- `train_entry.py`: deterministic-kernel setup and exact saved-initial-weight audit.
- `preflight.py`: initialization checks, 20-step repeat on GPUs0/1 for both modes, 300-step concurrent timing.
- `evaluation_smoke.py`: actual checkpoint standard/per-digit/paired evaluator smoke.
- `run.py`: two GPUs, whole GPU batch 7.5h cap including preflight, fail-stop, no retries.
- `worker.py`: 50k training followed by best/final evaluations.
- `evaluate.py`: standard41x256 and paired34x32 horizons per checkpoint.
- `summarize.py`: CPU count/config/checkpoint audit, compact archives and comparison plot.

Live root: `/mnt/raid0/RecursiveCompressor/experiments/logkv-aligned-rope-20260913/`.
Status: `campaign.json`, `AGENT_STATUS.json`; per-mode logs: `local-control/`, `aligned/`.
Weights/training logs: `exp/copying/<mode>-fixed10-20260913/`.
Source snapshot: `source/`, SHA256 record: `source_manifest.json`.
Stop supervisor with SIGTERM to stop its worker process groups.
No Selective, extra seeds, 16M evaluation or main merge is queued.

Completed and independently reviewed 2026-09-13. See [results and interpretation](../../logkv-aligned-rope.md).
`review.py` rechecks the original archive without changing the executed scripts or original hash index.
Selective continuation is a separate campaign: [protocol](../../logkv-aligned-selective.md).
