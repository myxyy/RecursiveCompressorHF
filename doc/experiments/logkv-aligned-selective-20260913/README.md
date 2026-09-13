# End-anchored RoPE Selective Copying experiment

Protocol: [../../logkv-aligned-selective.md](../../logkv-aligned-selective.md).
This is a separate continuation after the completed and audited Copying stage.

Both 50k runs and all 132 evaluation cells completed on 2026-09-13 at 21:36:04 JST.
All GPUs are released. Interpretation and limitations are in the protocol/report above.
The executed scripts and original result hash index remain unchanged.

- `common.py`: frozen source and shared initial-weight paths, Selective task binding, matched commands.
- `train_entry.py`: deterministic setup, actual task identity, exact saved-initial-weight check.
- `preflight.py`: 20-step repeats per mode on GPUs 0/1, concurrent 300-step timing.
- `evaluation_smoke.py`: real checkpoints, short horizons, independent memory/position replay.
- `run.py`: at most two GPUs; 7.5-hour cap including preflight; fail-stop, no retries.
- `worker.py`: 50k Selective training and best/final evaluation.
- `evaluate.py`: unchanged standard 33 horizons × 256 samples; actual positions, memory, predictions and margins.
- `summarize.py`: CPU replay/count/config/checkpoint audit; archive all 132 cells and plot.

Run with the repository `.venv/bin/python`. Run preflight, validate the elapsed-time
estimate and evaluator smoke, then launch `run.py`; it refuses missing checks,
existing runs, or work projected beyond the remaining stage budget.

Live root: `/mnt/raid0/RecursiveCompressor/experiments/logkv-aligned-selective-20260913/`.
Status: `campaign.json`, `AGENT_STATUS.json`; per-mode logs: `local-control/`, `aligned/`.
Weights: `exp/selective-copying/<mode>-fixed10-20260913/`.
Frozen source and initial weights are reused from the sibling Copying RAID root.
Stop the supervisor with SIGTERM to stop both worker process groups.
No extra experiments or main merge are queued.
