# Gated Mamba-2 branch + standard LogKV experiment records

See [the report](../../mamba2-logkv-gated.md). Serial hybrid was committed as `da7c8de` first.
The new branch preserves LogKV's embedding, CausalConv, blocks and untied head, and adds
an independent Mamba-2 feature branch through a zero-initialized per-channel tanh gate.

Started 2026-09-28 04:48:40 JST: Copying on GPU0, Selective Copying on GPU1.
One 50k-step training run per task, best/final x 41 horizons x 256 samples, total 164 cells.
Completed 2026-09-28 08:11:19 JST, including automatic audit (3h22m39s execution).
The CPU audit was rerun after completion; source files, weights and all 164 cells passed.
No additional seed, long-horizon extension or task was run. GPUs are released.
Copying best/final: 256/256 through T4096, 0/256 from T12288 onward (tested horizons).
Selective best: 256/256 through T1536; best/final: 0/256 from T8192 onward.
Best steps: Copying16300 / Selective45600. The report includes all four model comparisons.

- `implementation-checks.json`, `pytest.log`, `zero-gate-gpu-test.log`: 20 tests.
- `fullsize-smoke.json`: batch64/T2028 updates, finite gradients, gate opens from zero.
- `eval-batch-smoke.json`: finite actual-batch evaluation, peak allocated 11.83 GiB.
- Task `preflight.json`: source and upstream hashes, actual timing and deadline.
- Task `preflight-check.json`: 41 x 8 smoke recount, state formula, gate telemetry,
  and 41 x 256 paired LogKV targets / selective placement checks.
- Task `evaluation-smoke.json.gz`: preliminary evaluations, not final results.
- `launch.json`: supervisor PID, command, GPU assignments.
- `start-check.json`: 200/300 steps, gate updates, source hashes and matching preflight configs.
- `run_and_analyze.py`: fail-stop supervisor; `analyze_completed.py`: CPU audit and plots.
- `supervisor.json`: original completion status and timestamps.
- `review.json`, task `review.json`: completed 164-cell audit, including gate/weight agreement.
- Task `metrics.json`, `best.json.gz`, `final.json.gz`: metrics and full saved predictions.
- Task `train_log.jsonl`, `best.json`, `run_config.json`, `campaign.json`: training records.
- Task `comparison.png`, `gate-history.png`: four-model comparison and gate evolution.
- `diagnose_completed.py`, `completion-recheck.json`: current-source and raw-archive recheck,
  prediction diversity, digit accuracy and training instability (CPU only).
- `training-and-digits.png`: training intervals and T131072 per-digit accuracy.

Reproduce the CPU analysis from the repository root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  doc/experiments/mamba2-logkv-gated-20260928/analyze_completed.py \
  --root /mnt/raid0/RecursiveCompressor/experiments/mamba2-logkv-gated-20260928-run
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  doc/experiments/mamba2-logkv-gated-20260928/diagnose_completed.py \
  --root /mnt/raid0/RecursiveCompressor/experiments/mamba2-logkv-gated-20260928-run
```

Large artifacts and frozen sources:
`/mnt/raid0/RecursiveCompressor/experiments/mamba2-logkv-gated-20260928-run/`.
The post-run audit compares standard LogKV, standalone Mamba-2, serial hybrid and gated
hybrid on paired examples. Parameter counts and initialization are not identical.
Each task records the 512 effective gate scales every 100 steps; magnitude is not causal importance.
