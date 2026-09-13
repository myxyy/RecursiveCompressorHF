# Fixed-M10 combined-no-decay experiment artifacts

See [the experiment report](../../logkv-fixed10-combined.md) for status and protocol.

- `run.py`: two-GPU launcher, bounded to training plus the standard horizon evaluation.
  Frozen model/training source: commit `24b360cf85712fde3ee7a2da4d61e2eb45350a51`.
- `summarize.py`: after both tasks finish, validates run settings, all 500 training
  records per task, best selection and all 164 evaluation cells; archives raw data
  and writes comparisons with the historical fixed-M10 phase2 results.
- Models and optimizer-independent checkpoint files remain outside Git at
  `/mnt/raid0/RecursiveCompressor/experiments/logkv-fixed10-combined-20260908`.

From the repository root, after the run completes:

```bash
.venv/bin/python doc/experiments/logkv-fixed10-combined-20260908/summarize.py
```

The launcher refuses to overwrite existing runs. To repeat the experiment, change
its `ROOT` and `NAME` constants to unused locations/names and verify `SOURCE` points
to a clean worktree at the recorded commit. Do not launch a repeat or a longer
evaluation stage without observing the user's GPU/time confirmation constraints.


The approved boundary follow-up is also complete. `boundary.py` evaluates the
Copying final checkpoint at T=4064..4112 with paired memories; `check_boundary_batch.py`
checks the batch-size change at T4076/4077. `boundary/` stores all predictions,
counts, logs and the plot. These GPU scripts refuse to overwrite their outputs;
use a separate `OUT` for a repeat and follow the user's stage confirmation rules.
To validate the existing results and regenerate the plot without GPU use:

```bash
.venv/bin/python doc/experiments/logkv-fixed10-combined-20260908/summarize_boundary.py
```
