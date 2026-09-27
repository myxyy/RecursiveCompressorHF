# Mamba-2 + LogKV hybrid experiment records

See [the report](../../mamba2-logkv-hybrid.md) for architecture, controls and status.
This is one independent 50k-step run per task: Copying on GPU0 and Selective Copying on GPU1.
No automatic 16M/1B extension or extra seeds are queued.

- `implementation-checks.json`, `pytest.log`: 13 tests including official initialization,
  CPU/GPU chunk/gradient parity, nonmutation, state storage, serialization and generation.
- `fullsize-smoke.json`: two updates at batch64 and T2028 with finite gradients.
- `task-validation.json`: all 41 x 256 targets and selective positions match LogKV controls.
- `preflight-attempt1.json`: preserved unsuccessful initial preflight (allocator fragmentation).
- Task subdirectories: final 300-step timing, source hashes, 41 x 8 preliminary evaluation checks.
- `run_and_analyze.py`: two-task fail-stop supervisor; `analyze_completed.py`: CPU post-run audit.
- `launch.json`: added when the main training starts.

Large files, frozen sources and checkpoints:
`/mnt/raid0/RecursiveCompressor/experiments/mamba2-logkv-hybrid-20260928-run/`.
Both task results are audited against archived LogKV and Mamba-2 controls after completion.
Comparisons are matched by task/data, not parameter count (hybrid 9.22M, LogKV 5.79M, Mamba-2 3.45M).

Main training started 2026-09-28 01:14:16 JST on GPU0/1. Estimates including margins:
Copying 4.70h, Selective Copying 4.74h. Target completion around 06:00 JST;
hard deadline 08:10:33 JST. Both tasks completed before the deadline; see the completion notes below.
`start-check.json` confirms 100/200 steps, frozen sources and matching preflight configs.
`eval-batch-smoke.json` verifies finite outputs at actual evaluation batch sizes
(253 examples at T2048, 63 at T8192), peak allocated 10.83 GiB.
Both tasks use `PYTORCH_ALLOC_CONF=expandable_segments:True`; batch/data settings are unchanged.

Completed 2026-09-28 04:29:24 JST, including CPU analysis; GPUs released.
`supervisor.json` records completion. `review.json` audits 164 cells; each task folder
contains `metrics.json`, `comparison.png`, `best.json.gz`, `final.json.gz`, training logs,
and the selected checkpoint step. Copying best = final (step 50,000, identical weights);
Selective best = step 49,800, with training degradation in the final 200 steps.
`completion-recheck.json` records the later CPU verification of current/frozen source
hashes and byte-exact archived predictions; no extra GPU experiment was run.
`prediction-diagnostics.json` records per-digit errors and repeated predictions.
