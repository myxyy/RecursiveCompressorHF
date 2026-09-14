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

Started2026-09-14 14:39:11 JST on GPUs0/1; [launch evidence](launch.json) confirms identical initial weights and first100 training steps versus benchmarks. Estimated finish around19:40 JST; hard deadline22:02:35 JST. Results pending. Preparation `e7c4004`.
