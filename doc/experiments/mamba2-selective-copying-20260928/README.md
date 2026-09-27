# Mamba-2 Selective Copying experiment records

See [the report](../../mamba2-selective-copying.md) for protocol and status.
Large artifacts are stored at
`/mnt/raid0/RecursiveCompressor/experiments/mamba2-selective-copying-20260928-run/`.

- `task-validation.json`: CPU comparison of all 41 x 256 target strings, memory digits,
  and placement positions against the archived LogKV evaluation data.
- `preflight.json`: actual 300-step timing, source/dependency hashes, deadline.
- `evaluation-smoke.json`: preliminary checkpoint, eight examples at all 41 horizons;
  implementation check only, not the final performance result.
- `launch.json`: main-run supervisor PID, command and schedule.
- `analyze_completed.py`: CPU post-run audit and LogKV comparison plot; executed
  only after training and both evaluations succeed.

The task CLI defaults to Copying for backward compatibility. Selective Copying is explicit.
The campaign uses GPU0 only and stops after one seed with best/final evaluation.
Final metrics and figures are saved under `results/` when post-run analysis completes.

Completed 2026-09-28 00:51:32 JST, including CPU analysis. GPU0 is released.
Best checkpoint: step 46,800; final: 50,000. All 82 cells passed audit.
`results/metrics.json` and `results/comparison.png` contain the LogKV comparison;
`results/{best,final}.json.gz` preserve full predictions and margins.
`results/review.json` records the audit, and `completion-recheck.json` the subsequent CPU recheck.
`start-check.json` remains the historical 500-step snapshot, not the current status.
