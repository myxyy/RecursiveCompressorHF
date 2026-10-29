# 状態保存を省くLogKVの訓練用forward（2026-10-29）

LogKVとCausalConvのforwardを、全系列入力・初期状態なし・次回用hidden不要の
専用経路にした。対象は状態管理の削減のみ。attentionのgather、階層ごとの演算、
全階層共通softmax、activation checkpointingは既存の処理を使う。

## 実装

- `LogKV.forward`はQ/K/Vから圧縮階層を直接構築する。
  空キャッシュとの連結、未完成チャンクのclone、hiddenコンテナの構築を省く。
  端数チャンクと各レベルの参照規則は維持する。
- `_attend_contexts`へ既存のslot選択とattention呼び出しを抽出し、stepと共有する。
  各レベルの減衰・正規化・gate・self slotと再計算の設定を維持する。
- `CausalConvBlock.forward`は左側をゼロパディングして畳み込む。
  次回用の履歴テンソルを生成しない。空入力も扱う。
- `LogKVLM.forward`はpastなし・use_cache無効の場合にBlockのforwardを使う。
  pastがある呼び出しやキャッシュ付き生成はstep/predictを使う。
- パイプラインは元から`LogKVLMPipelineStage.forward → LogKVBlock.forward`を使うため、
  trainerの変更や追加フラグは不要。モデルのパラメータ名・形状・保存configも維持する。

## 正しさの確認

- CPU全テスト：447 passed / 21 skipped（CUDA専用テストは別途実行）。
- GPU 0で専用forwardと既存predictのテスト：102 passed。
- 新規テストはFP64独立参照との出力・勾配比較、C=2/3/4、再計算on/off、
  固定減衰・固定0・学習可能減衰・増幅・phase・norm・gate・self slotを確認。
  Convなし/幅1/幅4、入力長1/16/17/2047、LMとpipelineのforwardがstepを呼ばないこと、
  既存のキャッシュ経路も確認した。
- CUDAでFP32、BF16重み、FP32 master＋BF16 autocastの出力・勾配を確認。
- GPU 0/1で固定減衰・固定0それぞれの小規模パイプライン統合テストを実施。
  loss/勾配/global clip、全ラベル無効、保存・export・推論、層分割変更、
  中断再開時の重み・optimizer・RNGのbit一致を確認。
  このbit一致は**新実装内での連続実行と再開の比較**であり、旧実装とのbit一致ではない。

[検証記録](experiments/logkv-stateless-forward-20261029/review.json)。
小規模統合テストの重み・ログは
`/mnt/raid0/RecursiveCompressor/experiments/logkv-stateless-forward-20261029/`に保存した。
本学習や既存runの再開は行っていない。

## 速度とメモリ

RTX 3090、FP32 master＋BF16 autocast、C4・Conv4・gate/self slotあり・phaseなし・固定減衰。
入力長は実際のctx2048訓練と同じ2047。ランダム初期重みと合成入力を固定し、
同じモデルのLogKV/Conv forwardを従来のstep委譲へ戻して比較した。
各経路3回warmup後、順序を交互にして7組、各3回実行した平均時間の中央値を示す。
GPUを同期し、パイプラインでは全rankの最大時間を使う。

計測対象は**forward＋backward＋勾配クリア**。
パイプラインはtrainerと同じ`PipelineEngine`/1F1Bを使用し、入力転送・通信も含む。
optimizer更新、モデルロード、データ準備、生成、checkpoint保存は含まない。
したがって実際の長期訓練全体の短縮率を示す値ではない。

| 対象 | 従来 ms | 専用forward ms | 時間変化 | ピーク割当MiB（従来→専用） |
|---|---:|---:|---:|---:|
| 1 Block、d1024/FF3072、batch1 | 41.30 | 41.44 | +0.3% | 539.3 → 531.3 |
| 1 Block、d1024/FF3072、batch4 | 84.21 | 83.53 | −0.8% | 1690.3 → 1658.2 |
| 2GPU、4層、d512/FF1536、batch4 | 298.77 | 287.45 | −3.8% | rank0: 567.8 → 551.2、rank1: 390.7 → 382.6 |

単一Blockの速度差は小さい。小規模パイプラインでは約1.039倍の処理速度となったが、
大規模モデルや別の層分割への一般化には追加計測が必要。
パイプラインは2層ずつの分割、vocab1024、4 microbatches、accum1、
回答マスクを模した先頭1/3のラベル除外を使用した。

同じ入力でlossは一致したが、BF16の勾配はbit一致しない。
パイプラインの最大絶対勾配差はrank0で0.000397、rank1で0.0000191、
勾配全体の相対L2差はそれぞれ約0.742%・0.259%。
FP64では参照実装との厳密な許容誤差で出力・勾配が一致する。
演算グラフやテンソル配置の変更があるため、低精度訓練の過去軌跡のbit再現は保証しない。

[batch1の生データ](experiments/logkv-stateless-forward-20261029/block-b1.json) /
[batch4の生データ](experiments/logkv-stateless-forward-20261029/block-b4.json) /
[2GPUの生データ](experiments/logkv-stateless-forward-20261029/pipeline-2gpu.json)。
各JSONに全反復、rankごとのピーク割当、勾配差、環境、ソースhashを収録した。

## 再現

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -m pytest -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python -m pytest tests/logkv/test_logkv_forward.py tests/logkv/test_logkv_predict.py -q

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m benchmarks.benchmark_logkv_forward --target block \
  --dim 1024 --ff 3072 --batch-size 1 --output /tmp/forward-block-b1.json
# batch4は --batch-size 4 と別のoutputを指定する。

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  --module benchmarks.benchmark_logkv_forward --target pipeline \
  --dim 512 --ff 1536 --layers 4 --vocab 1024 --batch-size 4 --microbatches 4 \
  --output /tmp/forward-pipeline.json
```
