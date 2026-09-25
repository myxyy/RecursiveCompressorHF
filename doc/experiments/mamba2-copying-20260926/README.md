# Mamba-2 Copying experiment records

See [the report](../../mamba2-copying.md) for model fidelity, installation, protocol and current status.

- `preflight.json`: immutable source/dependency hashes, official upstream revision, measured runtime and deadline.
- `preflight-config.json`, `preflight-train.jsonl`: 300-step validation trajectory, not final training results.
- `evaluation-smoke.json`: 41 horizons through T131072 on eight examples with a preliminary checkpoint.
- `implementation-checks.json`: numerical/API checks and existing regression tests.
- `launch.json`: supervised main-run launch, GPU0 only.
- `start-check.json`: source integrity and initial progress; preflight replay is not bit-exact.

Full frozen source, model weights, logs and prediction-level evaluations are under
`/mnt/raid0/RecursiveCompressor/experiments/mamba2-copying-20260926-run/`.
The supervisor stops after this one Copying campaign; it does not start Selective Copying.


Completed 2026-09-26 01:03:57 JST (GPU0 released). `campaign.json` and `train_log.jsonl`
record completion. `results/metrics.json`, `results/comparison.png`, and `results/review.json`
contain the post-run CPU analysis. Full predictions and margins are in `results/{best,final}.json.gz`;
decompression reproduces the original RAID result files byte-for-byte.
Run `python doc/experiments/mamba2-copying-20260926/analyze_completed.py` with the experiment environment
to verify original artifacts, paired LogKV targets, source and weight hashes and regenerate the analysis.
The initial preflight/start records are preserved as historical snapshots, not current status.

`comparison-1b/` contains a separate comparison with the user-provided LogKV run at
`/mnt/raid0/RecursiveCompressor/exp/copying/1b/`: 67 horizons through T=1,073,741,824,
eight examples each. Its evaluation checkpoint and seed are unspecified in the aggregate results.
This is distinct from the 256-example historical LogKV control in `results/comparison.png`.
Mamba-2 remains evaluated only through T=131,072 (256 examples per horizon).
The directory includes copied LogKV aggregates/config, SHA256 provenance, PNG/SVG figures,
and CSV plotting data. Run `python doc/experiments/mamba2-copying-20260926/plot_billion_comparison.py`
to validate archived input hashes/metrics and regenerate the figure with matplotlib, without GPU work.
