# Fixed-decay LM training control,5000 steps

[Protocol and status](../../logkv-lm-fixed-decay.md).
Same frozen source, seed0 shared initial weights, six-GPU DDP, effective batch24,
training order and evaluation inputs as `../logkv-lm-learnable-decay-20260915/`.
Only omit `--learnable-decay` and change the run name. Fixed scalar bias arithmetic
under autocast differs from the learned tensor bias; this is an option-level comparison.

- `validate_initialization.py`: exact CPU equality of reconstructed shared untrained weights.
- `train_entry.py`: original trainer, verified initial weight fingerprint, progress telemetry,
  isolated sample RNG and control path. No model or optimizer-update changes.
- `evaluate.py`: same128 unconsumed rows and21 generation configurations as the completed control.
- `attention_valid.py`: fixed scalar-bias attention observation, only valid queries;128 rows.
- `summarize.py`: original learned-result hashes, fixed checkpoints, matching config/data,
  actual generated tokens and comparative loss/generation/attention metrics.
- `run.py`: six-GPU training -> GPU0 evaluation/attention -> CPU audit;7.5h total cap
  from preflight, no retries or queued follow-up training.

Completed2026-09-16 14:57:38 JST;4.53h execution/4.61h including preflight. GPUs released.
Post-completion CPU review passed:10 checkpoints,42 generation cases,4096 attention records.

Artifacts live under `/mnt/raid0/RecursiveCompressor/experiments/logkv-lm-fixed-decay-20260916/`.
Preserve scripts after launch and do not restart completed stages. Results and interpretation are in the linked protocol.

Started2026-09-16 10:25:40 JST on6 GPUs. Supervisor1198222/train1198223; preparation `e461e5e`.
Estimate5.45h,expected finish~16:00 JST; hard stop17:50:59 JST.
[Campaign status](campaign.json). Results and interpretation are in the linked protocol.


`analyze_completed.py` writes a separate `analysis/` without modifying frozen original results.
Reproduce with RAID checkpoints/tokenizer present:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python doc/experiments/logkv-lm-fixed-decay-20260916/analyze_completed.py
```

The original `complete-awaiting-review` campaign state is preserved as evidence.
All analysis in this review was CPU-only; no new training/evaluation campaign was launched.
