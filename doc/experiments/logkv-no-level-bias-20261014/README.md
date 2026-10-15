# Fixed zero level bias, 2026-10-14

Protocol and status: [../../logkv-no-level-bias.md](../../logkv-no-level-bias.md).

- Source: `ea64f2d` plus the recorded `implementation.patch`; the complete runtime
  source is frozen on RAID and SHA256-indexed in `source_manifest.json`.
- `no_level_bias=True`, `learnable_decay=False`; same 5,792,256 initial weights
  as the standard causal-convolution control. Initialization files have identical SHA256.
- `train_entry.py` loads and audits that initialization, then calls the frozen
  shared trainer with its explicit task module. No baseline weights are modified.
- `preflight.py` / `fullsize_smoke.py` check streaming, full-size training,
  20-step repeatability on GPUs0/1 and two 300-step benchmarks.
- `evaluation_smoke.py` replays predictions and task placements at three horizons.
  The first smoke attempt found a package import path missing in the CPU selective
  replay; it was fixed before training launch. Attempt logs are retained on RAID.
- `run.py` supervises two independent `worker.py` processes, each training 50k
  updates and then evaluating best/final. Whole-batch limit is 7.5h from first GPU
  preflight. Failure stops both worker groups. No automatic restart/new campaign.
- `summarize.py` audits every prediction, regenerates task targets/placements,
  verifies best selection and checkpoints, and archives results. `compare_baseline.py`
  checks paired examples against fixed-decay artifacts and makes a comparison plot.

Do not rerun historical preparation/launch scripts into the same directory.
Large artifacts: `/mnt/raid0/RecursiveCompressor/experiments/logkv-no-level-bias-20261014/`.
The automatic audit produces `results/`; full interpretation follows completion.
