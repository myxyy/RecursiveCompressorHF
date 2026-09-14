# Variable-memory evaluation of standard CausalConv LogKV

Protocol and status: [../../logkv-variable-memory.md](../../logkv-variable-memory.md).
Model merged into main at `654f311`; variable task implementation `90f13b1`.

- `common.py`: RAID paths, two-GPU mapping, frozen source verification and commands.
- `benchmark.py` / `preflight.py`: maximum-shape backward, real300-step runs,220-cell evaluation timing.
- `run.py`: bounded two-worker supervisor, fail-stop,7.5h including GPU preflight.
- `worker.py`: independent50k training and best/final evaluations; checks first300 intervals against benchmark.
- `summarize.py`: CPU RNG replay, prediction recount, training histogram/best checks and historical comparisons.

Runtime source is frozen under
`/mnt/raid0/RecursiveCompressor/experiments/logkv-variable-memory-20260914/source/`.
Task generation is byte-identical to the prior variable-M study; common initial weights have the same digest.
The new training enables strict determinism. No previous trained checkpoints are reused.

Preflight runs independently and never starts full training automatically:

```bash
.venv/bin/python doc/experiments/logkv-variable-memory-20260914/preflight.py
# Inspect preflight.json and the runtime estimate before the bounded campaign.
.venv/bin/python doc/experiments/logkv-variable-memory-20260914/run.py
```

Both refuse to overwrite completed stages/runs. No additional campaign is queued.
`campaign.json` under the RAID root records live status; per-task stdout is `<task>.log`.
Final raw NPZ observations and weights remain on RAID, with metrics and audit hashes archived here.

Started2026-09-14 14:39:11 JST on GPUs0/1; [launch evidence](launch.json) confirms identical initial weights and first100 training steps versus benchmarks. Estimated finish around19:40 JST; hard deadline22:02:35 JST. Completed2026-09-14 19:12:40 JST, all880 evaluation cells and CPU audit passed. Runtime4.56h (4.67h including preflight); GPUs released. Preparation `e7c4004`.


Post-completion interpretation is in the report linked above. Copying best/final share step50000 weights
and identical observations; M10/T131072 exact238/256, M16/T131072 exact1/256. Neither task has any
exact strings at M32/64/128 in the evaluated cells. Selective best48000 M10/T64 exact43/256
(final37/256); longest-horizon exact0/256. No additional runs are queued.

`results/` preserves the automatic audit snapshot. `analyze_completed.py` independently checks hashes
and recounts all880 NPZ observations on CPU, writing `analysis/` figures and review. Run from the repository root:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  doc/experiments/logkv-variable-memory-20260914/analyze_completed.py
```

All original scripts, frozen source, checkpoints and raw NPZ files on RAID are needed for the full verification.
