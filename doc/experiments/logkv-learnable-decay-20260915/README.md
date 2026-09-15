# Fixed10 CausalConv with learned per-head level decay

Protocol: [../../logkv-learnable-decay.md](../../logkv-learnable-decay.md).
Unmodified main model/CLI at `0e2e949`. One new configuration, existing `--learnable-decay`;
beta starts at log C, unconstrained, independently trained for Copying and Selective.
All common initial parameters match the completed fixed-decay CausalConv control.

- `validate.py`: fp64 fixed equivalence, signed-slope streaming, finite-difference gradients.
- `preflight.py` / `fullsize_smoke.py`: maximum shape, GPU0/1 bitexact repeats,300-step timing.
- `train_entry.py`: explicit task binding, frozen shared initialization and beta telemetry;
  observes AdamW updates without changing learning rates, clipping or parameter constraints.
- `evaluation_smoke.py`: real300-step checkpoint tests of observed targets/positions and scores.
- `run.py` / `worker.py`: two independent50k runs and best/final evaluations, fail-stop,
  7.5h total GPU cap including preflight, no retries or queued next campaign.
- `summarize.py` / `compare_baseline.py`: generator replay, counts, paired fixed-control comparison.
- `coefficient_audit.py`: all501 coefficient snapshots, best/final master and bf16 slopes.

Large files and frozen runtime live under
`/mnt/raid0/RecursiveCompressor/experiments/logkv-learnable-decay-20260915/`.
Do not overwrite/restart a completed stage. Preparation and launch status are in the linked report.
The standard configuration remains fixed decay; no LM, variable-M, zero-init, fixed amplification
or16M extension is queued. This checks the existing option, including its autocast bias-arithmetic difference.
