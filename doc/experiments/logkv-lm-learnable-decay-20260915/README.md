# Standard LogKV LM with learned per-head level bias, 5000 steps

[Protocol and status](../../logkv-lm-learnable-decay.md).
Frozen original `train_logkv.py` from `ce360e3`; 16 layers,128 slopes,6GPU DDP,batch4/accum1.
User explicitly authorized all six GPUs for this LM campaign. Confirm estimates >=8h;
whole campaign cap7.5h including GPU preflight; no further training queued.

- `common.py`: isolated RAID output/cache/control paths and full training command.
- `train_entry.py`: original trainer plus beta telemetry and RNG-isolated periodic samples.
- `evaluate.py`: checkpoint coefficient audit, unconsumed packed rows, inference-only beta interventions,
  actual level-attention mass and21 fixed generation cases.
- `summarize.py`: CPU checkpoint/history/finite-metric audits and portable plots/results.
- `run.py`: fail-stop 6GPU training -> 1GPU evaluation -> CPU audit supervisor.
- `preflight.json`, `benchmark_*`, `evaluation_smoke/`: actual30-step DDP/evaluator validation.

Large files live at `/mnt/raid0/RecursiveCompressor/experiments/logkv-lm-learnable-decay-20260915/`.
Once launched, preserve the source/scripts and launch hashes; never restart a completed campaign.
The normal CLI does not add experimental beta telemetry or isolate periodic sample RNG.

Started2026-09-15 23:02:55 JST, supervisor655438,training launcher655440. Preparation `5207743`.
Estimated completionSep16 04:30 JST; hard deadline06:26:50 JST.
See [campaign.json](campaign.json). Results pending.
