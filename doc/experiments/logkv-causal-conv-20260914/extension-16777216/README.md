# CausalConv Copying: T=16,777,216 extension

Authorized 2026-09-14 after review of the completed 41-horizon campaign.
This is evaluation only, on `logkv-causal-conv`; main is unchanged.

- Frozen runtime: `4bfee1f`, checked against the parent source manifest.
- Two-layer, no phase embedding, convolution width4; fixed memory length10.
- Copying `model_best` at step50000, identical to `model` (final).
- Weights SHA256: `0a42b83130217e513af2aa4d1000f185d27ca0c522a807a92d4789982af80468`.
- T=16,777,216; actual input length T+20=16,777,236. Eight examples, as in the historical 16M check.
- Original `task.make_batch`; seed12345, batch4. The same eight strings at each horizon.
  This resets the generator per horizon, unlike the prior 41-horizon sweep; the examples
  are not the same subset of that sweep at T131072.
- bf16 model weights and autocast, unmodified `model.step`, chunks8192, complete
  hidden-state carryover including convolution history. No blank skipping or recurrent-state approximation.
- GPU0 only. Preflight T131072 checks counts against the original evaluator;
  T1048576 measures time. Projection is measured seconds x16 x1.5 +60.
  Confirm before full execution if the projection is eight hours or longer.
- Save each memory, target, predicted digit, final answer logits and margins;
  recount independently after completion. Hash the frozen source and weights before and after.
- No training, Selective evaluation, additional architecture or main merge in this extension.

Outputs live under
`/mnt/raid0/RecursiveCompressor/experiments/logkv-causal-conv-20260914/extension-16777216/`.
The script refuses to overwrite a completed stage. From the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  doc/experiments/logkv-causal-conv-20260914/extension-16777216/evaluate.py preflight
# Inspect preflight.json and its runtime estimate before starting full.
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  doc/experiments/logkv-causal-conv-20260914/extension-16777216/evaluate.py full
```

Results pending.
