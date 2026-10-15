# Fixed zero level bias: Copying at T=16,777,216

User-authorized evaluation-only extension after the completed fixed10 campaign.
**Completed 2026-10-14 14:55:40 JST; CPU audit passed and GPUs released.**

At T=16,777,216 both best/final obtained 8/8 exact strings and 80/80 digits.
Minimum margins: best10.3046875, final6.546875. GPU evaluation took455.90s /463.55s
respectively (parallel supervisor wall time470.04s). See [review.json](review.json).

- best: step36,100 on GPU0; final: step50,000 on GPU1. Frozen trained weights, no retraining.
- Eight fixed random 10-digit memories, seed12345, batch4, chunk8192.
  Same eight memories as the historical fixed-decay CausalConv 16M evaluation.
- BF16 weights and BF16 autocast; the original task generator constructs the inputs.
  T is the memory horizon; the total input length is T+20.
- Each checkpoint passed T131,072 and T1,048,576 preflight (8/8 exact at both).
  At T131,072 counts also match an independent call to the standard evaluator.
- Source, checkpoint and config hashes are verified before/after evaluation.
  Predictions, answer logits and margins are retained for CPU recount by `review.py`.
- `stage.py` supervises two evaluation processes, stopping both on failure/timeout;
  no retry or further campaign is queued. Full-stage timeout is at least30min,
  or twice the preflight projection including margin, whichever is greater.
- Large artifacts/progress: `/mnt/raid0/RecursiveCompressor/experiments/logkv-no-level-bias-20261014/extension-16777216/`.

Eight examples are a small probe, not a claim of perfect accuracy for all strings or lengths.
Main report: [../../../logkv-no-level-bias.md](../../../logkv-no-level-bias.md).
