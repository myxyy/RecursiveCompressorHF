# Three-layer main LogKV without phase embeddings

Protocol: [../../logkv-no-position-3layer.md](../../logkv-no-position-3layer.md).

Adds one stacked LogKVBlock to the ongoing two-layer no-position control. All shared
parameters are copied from its **untrained initial** state; only layer 2 is newly
initialized. Both tasks load the same three-layer initial weights. No model source changes.

- `common.py`: frozen main source, paths, GPU 2/3 and fixed 50k-step task commands.
- `train_entry.py`: deterministic kernels, explicit task binding, initial-weight audits.
- `fullsize_smoke.py`: batch64, T2028, two optimizer updates; memory and finite-gradient check.
- `preflight.py`: matched initialization, 20-step GPU repeats and 300-step timings.
- `evaluation_smoke.py`: real benchmark checkpoints, independent task-data/count replay.
- `run.py`: two new workers, fail-stop, 7.5h cap including preflight, no extra stages.
- `worker.py`: each task's training and best/final evaluation, weight integrity checks.
- `evaluate.py`: 41 horizons through T131072, 256 examples each, per-digit predictions.
- `summarize.py`: CPU sample/count/configuration audit and compact results archive.
- `compare_depth.py`: paired two/three-layer predictions, including digit rescue/regression counts.
  If the independent baseline is not audited yet, records `pending-baseline` for later review.

New live root: `/mnt/raid0/RecursiveCompressor/experiments/logkv-no-position-3layer-20260913/`.
Reuses read-only `source/` and `initial_model/model.safetensors` from the two-layer campaign root.
Actual frozen files are identical to `3b0ce51`. Scripts and source are hashed at launch.
Status: `campaign.json`, `AGENT_STATUS.json`; task stdout: `copying.log`, `selective-copying.log`.
Weights: `exp/<task>/no-position-3layer-fixed10-20260913/{model_best,model}/`.

GPUs 0/1 and their original supervisor remain independent. This comparison alone is
authorized to use four GPUs in total. No seeds, positional variants or 16M evaluation are queued.
