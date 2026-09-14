# Main LogKV without phase embeddings

Protocol: [../../logkv-no-position-main.md](../../logkv-no-position-main.md).

Completed 2026-09-14 01:02:04 JST; both 50k runs and all 164 cells passed audits.
GPU 0/1 released. Results are in `results/`; interpretation is in the protocol report
and [the depth comparison](../../logkv-no-position-3layer.md). Original campaign
scripts and hash-indexed result artifacts are preserved unchanged.

One architecture, two independently trained tasks: fixed-M10 Copying and Selective Copying.
The frozen main runtime at `0932d8c` is identical to `3b0ce51` for all 27 captured files.
No model implementation changes. Earlier no-position results exist in the 2026-09-07 `none` runs;
`historical_reference.json` records those results separately from this deterministic rerun.

- `common.py`: frozen paths and fixed training command; explicit task binding.
- `train_entry.py`: deterministic setup and exact saved-initial-weight verification.
- `preflight.py`: shared initial state, 20-step repeats on both GPUs, 300-step timing per task.
- `evaluation_smoke.py`: both tasks/checkpoints on real 300-step models, with data/count replay.
- `run.py`: GPU 0/1 only; whole batch cap 7.5h including preflight; stop all workers on error; no retries.
- `worker.py`: one task's 50k training, best/final evaluation, checkpoint integrity check.
- `evaluate.py`: standard 41 horizons × 256 examples, true generated positions/targets/predictions/margins.
- `summarize.py`: CPU independent sample replay and count/config/weight audit; archive 164 cells.

Live root: `/mnt/raid0/RecursiveCompressor/experiments/logkv-no-position-main-20260913/`.
Status: `campaign.json`, `AGENT_STATUS.json`. Per-task logs: `copying/`, `selective-copying/`.
Weights: `exp/<task>/no-position-fixed10-20260913/{model_best,model}/`.
`LOGKV_TASK` identifies the task in training/evaluation subprocesses; `run.py` and preflight set it.
No additional runs or 16M evaluation are queued.
