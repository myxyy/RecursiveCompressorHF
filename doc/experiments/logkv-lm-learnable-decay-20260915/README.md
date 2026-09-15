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
CompletedSep16 03:33:21 JST,4.51h execution/4.61h preflight-inclusive.
See [campaign.json](campaign.json). All stages exit0; GPUs released.


Post-review correction: original `results/attention_mass.json` includes PAD queries in its
absolute trailing512 window. Do not use it to infer head roles. All other original metrics
remain valid. `attention_valid.py` profiled the same128 unseen rows onGPU0 in24.7s using
only real input/target positions; results and plots are in `analysis/`. No retraining.
`analyze_completed.py` verifies original hashes, five checkpoint coefficients, all generation
metrics and extended-prefix identity, then summarizes corrected attention.

CPU audit reproduction (RAID checkpoints/tokenizer required):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python doc/experiments/logkv-lm-learnable-decay-20260915/analyze_completed.py
```

Preserve original scripts and hash-indexed results; the two post-review analysis scripts
are separate from the completed campaign's frozen scripts.
