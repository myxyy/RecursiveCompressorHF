# CLAUDE.md

## Project Overview
Python ML project: a language model with a custom hierarchical-kv-compression architecture (**LogKV**, the current main line). The previous recursive-compression architecture (RecursiveCompressor) is retained as legacy. Uses HuggingFace (PreTrainedModel), PyTorch DDP, and uv for package management.

## Completed fixed-decay LM control (2026-09-16)
- User explicitly requested proposed fixed-decay comparison training and authorized all6 GPUs. One new5000-step run, same source `ce360e3`,327392256 parameters (128 fewer),standard16layers/d1024/ff3072/ctx2048/conv4/gate/self/phase-off. Same seed0, sampler/data, six-rank batch4/accum1, lr/warmup and sample RNG isolation as prior learned run. Do not resume any old run or change main model/CLI.
- Exact CPU shared-initialization comparison passed; actual training initialization fingerprint must match. Only argument changes versus learned control: learnable_decay/run_name. Existing fixed scalar vs learned fp32 tensor bias under autocast is a documented confound, not silently modified.
- Reuse same128 unseen rows and21 generation settings. Corrected attention uses only valid query positions on128 rows; never use old PAD-contaminated baseline attention. Source/checkpoint/baseline result hashes and prediction metrics audited automatically after completion.
- See `doc/logkv-lm-fixed-decay.md` and `doc/experiments/logkv-lm-fixed-decay-20260916/`. RAID folder matches experiment name. Preflight30 steps on6GPU plus evaluation/observer smoke; estimate~5.5h. Confirm >=8h; whole cap7.5h from GPU preflight. Train6GPU then evaluateGPU0 then CPU audit; stop at boundary, no retries/new campaign. STARTED2026-09-16 10:25:40 JST,supervisor1198222/train1198223,preparation `e461e5e`. Preflight and evaluator/valid-query observer passed; estimate5.45h,finish~16:00 JST,hard stop17:50:59 JST. COMPLETED2026-09-16 14:57:38 JST,4.53h execution/4.61h including preflight; all4 stages exit0,GPUs released.

- Completion analysis: fixed/learned heldout loss3.62275/3.61796,PPL37.44/37.26 (learned~0.48% lower); rank0EMA3.52558/3.52179. Three of four sources favor learned; MiniPile slightly favors fixed. One seed/autocast arithmetic confound; no significance claim.
- Temp0.7/1024 strong repetition counts4/9 both; Q4distinct0.539/0.461 (fixed/learned), EOS4/9 both. Temp1.0 strong repetition0/9 both,EOS8/9 vs6/9; shorter EOS is not quality proof. Fixed4096-limit lengths2332/4096/18, learned377/4068/18; both long story cases degenerate into place-name repetition. No good long-generation claim.
- Valid-query attention L>=2 mass24.12% fixed vs28.17% learned;+4.05pp,124/128 heads increase, corresponding-head correlation0.987. Same top3 compressed-reference heads L7H1/L9H5/L5H7. Roles already present in fixed model; learned increases allocation. This is attention mass, not causal contribution or repetition mechanism.
- CPU post-review checked10 checkpoints,42 generations,4096 attention records,source/scripts/results hashes. Artifacts preserved; new `analysis/` and `analyze_completed.py`. No new GPU run,training,main merge or default change. See `doc/logkv-lm-fixed-decay.md`.

## Completed 5000-step learned-decay LM campaign (2026-09-16)
- User requested `train_logkv.py --learnable-decay`,5000 new LM training steps plus head/output analysis. Explicit latest authorization permits all6 GPUs for this LM run; >=8h estimates still require confirmation. This supersedes the previous two-task campaign's no-LM scope only for this new campaign.
- Branch `adjust-attenuation`, frozen original source `ce360e3`. Standard d1024/H8/ff3072/16layers/C4/ctx2048,conv4/gate/self on,phase off;128 unconstrained beta initialized log4. No source architecture/CLI changes or main merge. Do not resume/overwrite user's separate32-layer run.
- Six-rank DDP,batch4/accum1,effective24,5000 steps from seed0; original Muon+AdamW lr2e-4,warmup1000. Archive all five1000-step checkpoints; beta every10 steps. Isolate periodic sampling RNG and experiment control/cache sentinels.
- Actual30-step preflight passed,2.768s/step,18.83GiB allocated on rank0; estimate5.42h including15% margin and1h eval. Evaluation smoke passed (observer output bitexact). Whole cap7.5h from benchmark start; fail-stop/no retries.
- Train on6 GPUs then evaluate onGPU0:128 unconsumed packed rows, learned vs inference-reset log4/zero loss,16x8 level-attention masses on four examples,18x1024 and3x4096 Japanese generations. Reset coefficients are not retrained baselines; row disjointness is not document deduplication; train loss is rank0, not DDP mean.
- See `doc/logkv-lm-learnable-decay.md` and `doc/experiments/logkv-lm-learnable-decay-20260915/`. Large artifacts/source/checkpoints in same-named RAID experiments folder. Preserve running scripts; stop at this campaign boundary. STARTED2026-09-15 23:02:55 JST, supervisor655438/train655440,6GPU. Preparation `5207743`; estimated finishSep16 04:30 JST, hard stop06:26:50 JST. COMPLETED2026-09-16 03:33:21 JST,4.51h execution/4.61h including preflight; all stages exit0,GPUs released.

- Completion review: final rank0 EMA3.5218; beta all128 positive and belowlog4,min0.6850/mean1.0215/max1.3295;121 heads still decreased in last1000 steps. Unseen-row weighted loss learned3.61796/reset-log4 3.63179/zero3.79552; inference interventions, not trained controls.
- Generation1024: temp0.7 EOS4/9,meanQ4distinct0.461,4/9 below0.5; temp1.0 EOS6/9,distinct0.852,0/9 below0.5. 4096-limit seed0 generations reuse exact prefixes: lengths377/4068/18, allEOS; only4068 case crosses training ctx and degenerates into place-name repetition. Do not claim long generation solved.
- Post-review found original four-row attention averages includePAD. Preserve original results but do NOT interpret those attention masses. Corrected profile in `analysis/attention_valid.json`: last<=512 real input+target positions from all128 same rows,64338 queries/head;GPU0 rerun24.7s,then released. Observer output bitexact on four sources. Layer1 self92.83%;L7H1 L>=2 mass73.21%;positive beta does not preclude strong compressed-memory attention. See `analysis/review.json`;five checkpoint hashes/history,all21 text/token metrics verified. No additional training/merge.

## Completed learned level-decay comparison (2026-09-15)
- STARTED2026-09-15 16:43:00 JST on GPUs0/1; supervisor306138, workers306150/306151. First100 metrics AND beta histories bit-match benchmarks; baseline config differs only run_name/learnable_decay/num_params. Branch `adjust-attenuation`, preparation `1762bfb`, frozen source `0e2e949`; main standard unchanged. Completed2026-09-15 19:33:38 JST, execution2.84h/preflight-inclusive2.93h. GPUs released; `launch.json` preserves launch evidence.
- User authorized existing `--learnable-decay` on standard main CausalConv, fixed M10/P0 Copying and Selective. Keep beta initialized at log C, unconstrained per layer/head; not zero-init. Frozen main source `0e2e949`, no model/CLI changes.
- Baseline: completed `logkv-causal-conv-20260914` fixed-decay runs; copy ALL shared untrained weights including convolution, add only16 slopes (5792272 params). Two independent50k tasks, batch64/accum1/T loguniform1..2028, same seeds/best selection/bf16-weight eval,164 cells through T131072 x256.
- GPU0/1 only; preflight passed,20-step GPU repeats bitexact,300 steps61.4s/57.8s, peak15.85 GiB. Estimated4.30h including preflight. GPU preflight started2026-09-15 16:37:56 JST; hard stop2026-09-16 00:07:56 JST (7.5h). Evaluation and coefficient-update smoke checks passed. All50k runs/164 cells completed; CPU post-review recounted328 baseline/new cells and verified frozen artifacts/weights/beta histories.
- `doc/logkv-learnable-decay.md` and `doc/experiments/logkv-learnable-decay-20260915/` record the protocol. All large artifacts under `/mnt/raid0/RecursiveCompressor/experiments/logkv-learnable-decay-20260915/`. Preserve frozen source and scripts once running; do not restart archived campaigns.
- Beta telemetry every100 updates; verify master/bf16 checkpoint slopes, sign and per-head changes. Existing learned-bias tensor changes autocast logit addition precision versus fixed scalar bias; document this confound rather than silently changing the option.
- Results: Copying best38600 gets256/256 at all41 horizons throughT131072; final50000 degrades (T3=35,T7=24,T131072=75),2594/2608 digit errors at digit9. Selective best42900 T64=248 vs94 control,T1024=190 vs10,T2048=158 vs14; final lower,T131072 exact0/0. Copying beta stays positive; Selective4/16 heads amplify. One seed/precision confound; no16M claim for these weights. See report and `analysis/review.json`; preserve original `complete-awaiting-review` campaign state and hash-indexed results.
- Standard remains fixed decay. No variable-M, zero-init, fixed amplification, LM run or16M extension queued. Stop after this two-task campaign and its CPU audit.

## Standard configuration and newly authorized variable-memory study (2026-09-14)
- COMPLETED2026-09-14 19:12:40 JST: both50k runs,880 best/final cells and CPU audit passed. Execution4.56h, preflight-inclusive4.67h; all GPUs released, nothing else queued. Copying best=final step50000, identical weights/data; M10 T131072 exact238/256, M16 exact1/256. M32/64/128 have no exact strings in any evaluated cell. Selective best48000, M10 T64 exact43/37 (best/final); T131072 exact0/0, digit59.49%/53.24%. Post-completion hashes/recounts passed; see `doc/logkv-variable-memory.md` and `analysis/review.json`.
- MERGED locally into main at `654f311`; variable task code `90f13b1`. 159 model tests and14 variable-task tests passed, plus CLI default/opt-out checks.
- Historical variable campaign preparation: 300-step benchmarks and440 evaluation cells with CPU RNG/position replay passed. Estimated5.08h including preflight; GPU0/1 only, maximum allocated8.98 GiB each. Deadline2026-09-14 22:02:35 JST. See `doc/logkv-variable-memory.md`; raw/source/checkpoints under `/mnt/raid0/RecursiveCompressor/experiments/logkv-variable-memory-20260914/`. Preserve frozen source and running scripts; never restart a campaign from historical launch examples.
- User authorized merging CausalConv into main as standard after the successful16M check, then variable-memory Copying/Selective training. This supersedes historical no-merge/no-further-experiments scope below.
- New-training CLI defaults: conv width4, gate and self slot on, phase off. Low-level config/block defaults stay compatible with old checkpoints. Disable via `--conv-kernel-size 0 --no-gated-attention --no-self-slot`.
- Variable study: reuse prior M={10,16,32,64}, P0..63, T loguniform1..2028, 50k steps, microbatch32 x2, validation every2000, best/final220 cells each x256. One new standard architecture only, two independent tasks on GPUs0/1, benchmark before launch; confirm >=8h and stop at experiment boundary.

## Historical causal-convolution development and fixed10 campaign (2026-09-14)
- COMPLETED additional user-authorized 16M Copying evaluation: T=16777216 exact8/8, digits80/80, minimum logit margin10.125. Fixed M10, seed12345, same step50000 checkpoint; no retraining. GPU0 only, 456.83s, completed2026-09-14 14:20:54 JST and released. Preflight T131072/T1048576 also8/8 on the same strings. See `doc/experiments/logkv-causal-conv-20260914/extension-16777216/` and its `review.json` (CPU recount/RNG/source/weight audits passed). This supersedes the earlier no-16M scope for this completed extension only; no Selective16M, further experiments or main merge.
- User authorized a new branch and implementation/experiments for causal convolution before each LogKVBlock attention. Branch `logkv-causal-conv` starts at main `d707675`; main itself stays unchanged. This new authorization follows the completed positional/no-position studies below.
- `conv_kernel_size=0` (default) preserves existing model parameters and attention hidden format; 4 enables token-stream RMSNorm -> depthwise causal Conv1d -> SiLU residual before attention. Each block caches only the last k-1 normalized inputs across chunk/step boundaries. No per-compression-level convolution. Keep the refined attention slot layout and fixed level decay.
- Only new experiments authorized: two-layer, no phase embeddings, conv width4, fixed M10 Copying and Selective, independently trained 50k, best/final 41 horizons through T131072 x256. Shared initial parameters match the completed two-layer no-position baseline's untrained weights; new conv parameters only. Baseline results already exist; do not retrain it or restart archived runs.
- Maximum two GPUs (0/1); benchmark first, confirm if a proposed batch is expected to take >=8h, whole new batch capped at 7.5h including GPU preflight. No additional seeds, widths, per-level variants, three-layer models or 16M extension are queued.
- Core tests: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest test_logkv.py test_logkv_lm.py test_logkv_conv.py -q` (159 passed). Convolution's 3D weights must use AdamW, not Muon; LM optimizer now requires ndim==2 for Muon.
- Frozen runtime is `4bfee1f` (28 files). GPU preflight started 2026-09-14 11:00:42 JST; batch64/T2028 fits (15.85 GiB allocated), GPU fp64 streaming error 5.6e-17, 20-step task repeats bitexact across GPU0/1. 300 steps 61.7s/57.7s project 4.31h including margin/evaluation/preflight; hard stop 2026-09-14 18:30:42 JST.
- Started 2026-09-14 11:04:31 JST on GPUs0/1, supervisor PID 2877069, workers 2877070/2877071. Both first 100-step intervals bit-match their benchmarks. Preparation commit `dab79f2`; source `4bfee1f`; main still `d707675`. Completed 2026-09-14 13:55:28 JST, all worker commands exit 0. Runtime 2.85h (2.91h including preflight), within the hard stop. GPUs released; no extra stages queued.
- See `doc/logkv-causal-conv.md` and `doc/experiments/logkv-causal-conv-20260914/`. Completion review passed: Copying 256/256 exact at all 41 horizons through T131072 (10,496 examples); best step50000 equals final weights and data. Selective T64 exact94/112 vs baseline0/1 (best/final, each256); T131072 exact0/0, digit41.05%/48.24%. No 16M or additional seed evaluation. `analysis/review.json` records post-completion hash checks and recounts of 328 baseline/conv cells. All large artifacts live under `/mnt/raid0/RecursiveCompressor/experiments/logkv-causal-conv-20260914/`. Preserve frozen scripts/source/prerequisite JSON and hash-indexed original results; the completed campaign state `complete-awaiting-review` is retained as historical evidence.

## Current direction and experiment archive (2026-09-13)
- User explicitly paused positional-encoding tuning and requested documentation/experiment-code-only cherry-picks into main. This authorizes the archive integration even though replacement encodings did not pass the 16M Copying gate; it does not authorize merging their model implementation.
- Main's model, configuration, training entry points, baseline task suites, dependency files and normal tests remain exactly as at `658f63e`. Use that existing architecture for subsequent training.
- All planned positional comparisons are complete and GPUs are released. Do not restart archived campaigns merely because an old report contains a launch command or historical continuation permission.
- Reports and historical launch/analysis scripts live under `doc/`. `doc/logkv-experiments.md` indexes their frozen source commits and reproduction requirements. Experimental APIs described there are not present in main.
- `doc/experiments/position-code-archive-20260913/experiment-code.tar.gz` preserves the variable-length study code, experimental Copying CLI and model tests, without collecting/running those tests against main. Use the appropriate original commit in a separate worktree for reproduction. The source branch `logkv-aligned-rope` remains intact.
- Retain the user's power constraints: continuous work on at most 2 GPUs is authorized; confirm before a proposed experiment/batch expected to take 8 hours or longer. Earlier campaign-specific 3-GPU permissions are not a general increase. Respect meaningful experiment boundaries and keep Copying/Selective as separate training/checkpoint/evaluation tasks.
- No training is started by this archive integration. Checkpoints and frozen source directories stay under `/mnt/raid0/RecursiveCompressor/experiments/`; archived absolute paths refer to that machine.

## Authorized no-position baseline rerun (2026-09-13)
- User requested fixed-M10 Copying/Selective evaluation with phase embeddings disabled on main's refined architecture. Historical `none` runs already exist in the 2026-09-07 relative-bias study; record them as prior data, not as an unmeasured condition.
- This new main-only rerun uses frozen source `0932d8c`; all 27 captured runtime/dependency/task files are identical to `3b0ce51`. Keep gate/self slot/fixed level decay and model size; no RoPE or added positional corrections. Two independent 50k task models, best/final 41 horizons through T131072 x256 (164 cells).
- Started 2026-09-13 22:18:53 JST, supervisor PID 2086202, GPUs 0/1 only. Both first 100-step intervals match their benchmarks exactly. Hard deadline 2026-09-14 05:43:53 JST. Deterministic kernels, shared initial weights, 20-step repeat per task bitexact across GPUs. Measured 300-step benchmark predicts 4.11h including 15% margin, one hour evaluation and preflight. Whole batch cap 7.5h from preflight start; stop all on failure, no retries or extra runs/16M.
- All preparation changes are documentation and experiment scripts under `doc/`; main model/trainer remain unchanged. See `doc/logkv-no-position-main.md` and `doc/experiments/logkv-no-position-main-20260913/`.
- COMPLETED 2026-09-14 01:02:04 JST: both 50k runs and 164 best/final cells passed audits; GPUs 0/1 released. Runtime 2.72h (2.80h including preflight). Best Copying T64=31/256, T131072=1/256; best Selective both 0/256. No continuation queued.

## Architecture
### Authorized three-layer no-position comparison (2026-09-13)
- User explicitly authorized two additional GPUs for three-layer Copying/Selective alongside the running two-layer control. For this comparison only, GPU 0/1 remain the existing control and GPU 2/3 run the new task models (four total). This is not a general increase for later work.
- Only num_layers changes to 3 (stacked LogKVBlocks, not a cap on compression depth), 8,673,792 parameters. Same main source, fixed M10, 50k steps and best/final 41 horizons through T131072 x256. Shared parameters match the saved untrained two-layer initial state; only the extra block is newly initialized. No trained checkpoint transfer.
- Full batch64/T2028 smoke and both task repeat/evaluation checks passed. Preflight began 2026-09-13 23:33:24 JST; projected 6.29h including margins/evaluation, hard stop 2026-09-14 07:03:24 JST. Separate fail-stop supervisor; do not alter the ongoing two-layer scripts/source or their deadline. No further variants/seeds/16M are queued.
- Started 2026-09-13 23:40:44 JST, supervisor PID 2166116, workers 2166117/2166118. Both first 100-step intervals match their respective benchmarks exactly. Expected completion around 2026-09-14 06:00 JST. Preparation commit `3cd0790`; launch evidence in the experiment folder's `launch.json`.
- See `doc/logkv-no-position-3layer.md` and `doc/experiments/logkv-no-position-3layer-20260913/`. Historical positional degeneracy was diagnosed on the overlapping layout; do not claim it is a proof for current main. Compare paired digit errors and acknowledge parameter-count growth and one seed.
- COMPLETED 2026-09-14 04:14:42 JST: both 50k runs and 164 cells audited, all GPUs released; runtime 4.57h (4.69h including preflight). CPU review verified all 328 cells, paired samples, frozen source/scripts/artifacts and checkpoint hashes. Reports/results committed on main; model unchanged. Do not restart these campaigns.
- Three-layer best Copying T64=237/256, T192=234/256, T256 onward=0/256; long digit accuracy worsens (T131072 71.91% to 53.75%, final 70.98% to 41.72%). Selective digit accuracy improves at all 41 horizons for both best/final; T131072 best 21.72% to 62.38%, exact 0 to 1/256. At T256 three-layer best always outputs target digit 2 at output digit 3 (226 errors, 30 coincident matches); same architecture final gets digit 3 correct in 249/256. This collision is not universally forced by architecture. Hidden/logit equality unmeasured; one seed and capacity confound remain.
- CPU analysis code and plots are in `doc/experiments/logkv-no-position-3layer-20260913/analyze_completed.py` and `analysis/`. Original campaign scripts and hash-indexed result artifacts stay frozen. Proposed denser boundary/hidden-state diagnostics are not executed or queued.

### LogKV (main)
- `logkv.py` - Core module. Per level i (sub-unit = C^i tokens), each query attends only to completed sub-units in its current block (c < j); these disjoint intervals partition the entire past. All levels share one softmax (at most C−1 slots per level). See `logkv-refine.drawio.png` and `doc/logkv.md` §6.17. Compression is attention pooling with the chunk-last query. Has `forward`/`step`/`predict` (fp64 machine-precision equivalent) plus `LogKVBlock` (pre-norm attention+FFNSwiGLU) and options: `phase_emb`/`phase_levels`, `gated_attention`, `self_slot`, `learnable_decay`, `kv_norm`, `v_norm_only`, `level_amplify`.
- `logkv_lm.py` - LogKVLM language model (PreTrainedModel + generate; `past_key_values` carries the opaque per-layer hidden list).
- `configuration_logkv.py` - LogKVConfig.
- `train_logkv.py` - **DDP data-parallel** training (the model fits on one GPU). Muon + AdamW, bf16 autocast, control.cmd, `--resume latest` (skips consumed data, absolute `--max-steps`, EMA carry-over), periodic Japanese sample generations to `samples.log`.
- `predict_logkv.py` - Text generation for LogKV checkpoints.
- `exp/copying/`, `exp/selective-copying/` - Copy Memory Problem / Selective Copying suites (`--arch logkv` supported; selective wraps copying via task-module injection).
- `doc/logkv-experiments.md` - Index of all experiment reports, including experimental-branch positional encodings. Those reports do not imply their optional implementations are present on main.
- `doc/logkv.md` - **The design/experiment record for LogKV. Read this first for any LogKV work.**

### Shared
- `dataset.py` - Data pipeline with memmap caching. Tokenizes HF datasets, packs short documents into context-length sequences.
- `predict.py` / `predict_stream.py` - Generation / interactive streaming REPL. `_load_model` picks the architecture from config.json's `model_type` ("logkv" → LogKVLM); also detects legacy pipeline checkpoints (`full_model.pt`).
- `chat_server.py` - Chat web UI (legacy-model era; decaying repetition penalty, reset/interrupt).

### Legacy (RecursiveCompressor)
- `recursive_compressor.py`, `recursive_compressor_lm.py`, `recursive_compressor_lm_pipeline.py`, `configuration_recursive_compressor.py`, `train_pipeline.py` (6-GPU pipeline parallel, Schedule1F1B). History: `doc/copying-memory-branch-changes.md`; full experiment logs under `doc/instruction-for-claude/`.

## Commands
```bash
uv sync                                                # Install dependencies
uv run pytest test_logkv.py test_logkv_lm.py -v        # LogKV tests
uv run pytest test_lm.py -v                            # Legacy tests

# LogKV standard-config training (DDP, 6 GPUs)
uv run torchrun --nproc_per_node=6 train_logkv.py --run-name <name> \
    --conv-kernel-size 4 --gated-attention --self-slot

uv run python predict_logkv.py --model-dir $DATA_DIR/checkpoints_logkv/<name>/checkpoint-<step>/model \
    --max-new-tokens 1024 --temperature 0.7 --top-p 0.9
uv run python predict_stream.py --model-dir /path/to/checkpoint --temperature 0.7 --top-p 0.9

uv run torchrun --nproc_per_node=6 train_pipeline.py   # legacy pipeline-parallel training
```

## Training Control
```bash
just pause / just resume / just save-and-exit   # writes control.cmd (pause keeps GPUs allocated but idle)
```
Resume flags must match the run's original flags (model structure comes from the checkpoint's config.json).

## TensorBoard
`train/loss`, `train/grad_norm`, `train/lr` per step: LogKV runs under `$DATA_DIR/tensorboard/logkv-{dataset_type}/{run}/`, legacy under `$DATA_DIR/tensorboard/{dataset_type}/`.
```bash
uv run tensorboard --logdir $DATA_DIR/tensorboard/
```

## Environment
- `.env` file sets `DATA_DIR` (datasets, checkpoints, memmap caches)
- Default: `DATA_DIR=./data`; Production: `DATA_DIR=/mnt/raid0/RecursiveCompressor`
- Hardware: 6x RTX 3090 (24GB VRAM each), 256GB RAM

## Key Design Decisions — LogKV
Details and evidence live in `doc/logkv.md`; summary:
- **Refined layout (2026-09-06)** removes overlapping previous-block slots. Matched 50k-step experiments (one seed, phase2 + gated + self slot) preserve perfect Copying through T=131072 (41 horizons, n=256, best/final), with an additional 8/8 at T=16777216, but reduce Selective Copying accuracy (T=64 best string: 30.5% → 17.6%). See `doc/logkv-refine-experiments.md`. LM quality and the LM throughput measurements below still belong to the overlapping layout. Weight shapes/config flags are unchanged, so old checkpoints load with new attention semantics. Restart generation with hidden=None; old runtime hidden states are incompatible.
- **Standard config**: fixed level decay (−i·log C) + width4 token-stream CausalConv (phase off) + multi-head + gated attention + self slot (the query's own k/v as one extra slot = standard causal-mask semantics; loss-neutral, gives the softmax an "attend to nothing" option, stabilizes grad_norm ~1.14→0.74). `kv_norm`/`learnable_decay`/`level_amplify`/`v_norm_only` exist as options but are NOT standard (each was tested and rejected for the LM: kv_norm caps the key-norm retrieval margin, learnable decay and amplification worsen temp-0.7 repetition, v_norm alone helps little).
- **Level decay** originally corrected cross-level multiplicity and improved topic fixation. It is retained as a coarse-level penalty in the refined layout, where slots no longer overlap; its former multiplicity rationale no longer applies. The original 16.7M-token Copying result is in §6.13; a separate refined-layout probe is recorded in `doc/logkv-refine-experiments.md`. The variable-memory study in `doc/logkv-position-study.md` ablates decay only with the new combined positional scheme. Decay ablations for standard phase2 and LM quality remain untested.
- **Variable-memory positional study (2026-09-07–08)**: code `24b360c` on `logkv-position-study`; main stores documentation and evaluation data. All 12 runs completed 50k steps and best/final evaluation. Exact Copying remains unsolved; relative K/V improves Selective M10/T64, and combined/no-decay improves very short-horizon M16/M32. One seed, different training conditions from the earlier fixed-M10 studies. See `doc/logkv-position-study.md`.
- **Phase embedding** (base-C digits of the absolute position, small period) breaks the positional degeneracy inside runs of identical tokens (multi-scale windows coincide there, making "how many so far" uncountable). Longer periods aliased beyond the training range and hurt extrapolation in earlier configurations; period 16 was the robust choice there. The variable-memory study documents its limits for longer payloads and does not establish arbitrary-length exact copying.
- **step()/forward()/predict() equivalence** is the core invariant, tested at fp64 <1e-12 against an independent `reference_forward` in `test_logkv.py` (forward delegates to step, so the test oracle must stay independent). Any semantic change (biases, norms, gates) must be mirrored in the reference.
- **Hidden format**: `(levels, offset)`; `levels[i] = [cur_q, cur_k, cur_v]` holds only the unfinished chunk (<C entries), with heads folded into the batch dim. Offset reconstructs absolute positions (phase, block bases). Retained slices are cloned to avoid pinning full-segment storage during inference. O(C·log L·d) memory.
- **Online softmax + activation checkpointing**: the whole attention pass is one non-reentrant checkpoint region; autograd keeps only per-level contexts (~1.33·L·d) instead of (L, C·levels, d) slot gathers. bf16-weight inference (no autocast) needs the softmax weights cast to the value dtype (fixed; see test).
- VRAM/throughput at d1024/8H/16L/ctx2048 (~310M params): batch 4/GPU ≈ 16 GiB, ~16-17k tok/s total on 6 GPUs (~4h per 5000 steps).

## Key Design Decisions — data pipeline & training (shared)
- **Tokenizer**: `elyza/ELYZA-japanese-Llama-2-7b-fast`. `[INST]`, `[/INST]` are Llama-2-style plain text markers (not special tokens).
- **Data format** (Llama 2 style): Documents: `<s>text</s>`. Conversations: `<s>[INST]q1[/INST]a1</s><s>[INST]q2[/INST]a2</s>...` (each turn BOS+EOS-wrapped).
- **Pretrain chunking + packing** (`_build_memmap_packed`): `[BOS] + tokens + [EOS]` split into context_length chunks (first chunk has `<s>`, last has `</s>`, continuations unmarked); chunks packed to fill samples, PAD-filled; loss on all non-PAD positions.
- **Instruct conversations** (`_build_memmap_conversations`): prompt `<s>[INST]q[/INST]` and answer `a</s>` tokenized separately for an exact loss-mask boundary; answer-only loss mask in a parallel `.mask` memmap; no cross-conversation packing.
- **Memmap caching**: uint16 memmaps under `$DATA_DIR/hf_cache/mmap/ctx{context_length}/`; per-source version suffix (pretrain `_v5`, instruct `_v6`) — bump a suffix to force rebuild. `prefault=True` warms the OS page cache on rank 0 (shared across ranks).
- **All-PAD-label NaN guard**: `_pack_chunks` enforces `MIN_CONTENT=2`; loss functions also return `logits.sum() * 0.0` for all-PAD (micro)batches (0/0 CE = NaN otherwise).
- **Mixed precision**: fp32 master weights/optimizer state, bf16 autocast forward/backward. RMSNorm computes in fp32 but outputs bf16. CE loss gets `logits.float()`.
- **Optimizers**: Muon for 2D hidden Linear weights (`adjust_lr_fn="match_rms_adamw"`), AdamW for embedding/head/biases/norms and non-2D params (`_ADAMW_ONLY_KEYWORDS` includes `phase_emb`; Muon rejects non-2D tensors).
- **Sampler shuffle**: `DistributedSampler` seed=0 + `set_epoch` gives a reproducible order; train_logkv resume skips consumed samples via a SkipSampler.
- **Legacy pipeline notes** (train_pipeline.py): Schedule1F1B loss collection via `losses=[]`, per-stage checkpoints + reconstructed `full_model.pt`, `STAGE_LAYER_SPLIT` for VRAM balance, per-step ReduceLROnPlateau on EMA loss. Cache building happens before `init_process_group` (sentinel file); all ranks barrier after checkpoint saves.

## Debugging Guidelines
- After modifying LogKV, run `uv run pytest test_logkv.py test_logkv_lm.py -v` before committing; the fp64 reference/step/predict equivalences must stay <1e-12.
- When modifying the data pipeline (packing, collation), add shape/length assertions; all packed sequences must be exactly context_length (`(seq + [PAD] * context_length)[:context_length]`).
- **NaN debugging**: training is deterministic with `seed=0` + `set_epoch(epoch)`, so resuming reproduces the same NaN at the same step. Loss=NaN with finite GradNorm typically means all-PAD labels (0/0 CE), not bad logits.
- Generation-quality checks use the repetition metric suite (temp 0.7/1.0 × 3 seeds × 1024 tokens: EOS rate, Q4 bigram distinct ratio, heavy-repetition count, longest same-char run) — see doc/logkv.md §6.6 for baselines.
- Legacy tests `test_step_split_consistency`/`test_predict_matches_forward` use loose tolerances (`atol=5e-3`) due to data-dependent xs propagation.

## Current Model Parameters (LogKV standard)
- d_model=1024, num_heads=8, d_ff=3072, chunk_size=4, num_layers=16 (~310M params)
- phase_emb=False, conv_kernel_size=4, gated_attention=True, self_slot=True; decay fixed at log C
- context_length=2048, lr=2e-4 (linear warmup 1000), DDP batch 4/GPU × 6 GPUs
- mixed precision: fp32 master / bfloat16 autocast
- (Legacy pipeline model: d_model=2048, num_heads=16, d_ff=6144, num_layers=16, lr=5e-5)

## Datasets
Selected by `--dataset-type` in `train_logkv.py` / `train_pipeline.py`:
- `pretrain` (documents only):
  - `wikimedia/wikipedia` (20231101.ja, 20231101.en)
  - `hotchpotch/cc100-ja-documents`
  - `JeanKaddour/minipile`
- `instruct` (conversations only):
  - `shi3z/ja_conv_wikipedia_llama2pro8b_30k`
  - `shi3z/ja_conv_wikipedia_orion14B_100K`
  - `HuggingFaceH4/ultrachat_200k`
