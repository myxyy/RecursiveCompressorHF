# LogKV全系列forwardのK/V共有（2026-10-29）

[状態を保存しないforward](logkv-stateless-forward.md)をコミット`9610511`に保存し、
その上で階層ごとのattentionを行列積にまとめた。変更ブランチは`logkv-grouped-forward`。
訓練CLIの追加引数は不要。

## 計算方法

レベルiのサブユニット長をS=C^iとすると、CSトークンごとのクエリ群は
同じC個のサブユニットから、位置jより前のc<jだけを参照する。
最後のサブユニットは同じブロック内から参照されないため、K/Vは最大C−1個でよい。

従来は各クエリへK/Vをgatherし、`(B*H, L, C-1, head_dim)`を展開していた。
新経路ではQをブロック単位にまとめ、各ブロックのK/Vを共有してQKとPVを行列積で計算する。
スコア行列は概ね`(B*H, L, C-1)`であり、全系列同士の密なattentionは作らない。

- 小さな端数はQをパディングして一括処理する。追加長を元の25%以下に制限する。
  それを超える場合は最後の不完全ブロックだけを別計算し、上位階層での大きなQ展開を防ぐ。
- 各レベルのmax、分母、value積を算出し、既存のonline softmaxへ渡す。
  レベルごとに独立したsoftmaxで出力を確定する変更ではなく、全階層＋self slotの共通正規化を維持する。
- FP32以上の統計量、value dtypeでの積、固定減衰・固定0・学習可能減衰などの設定を維持する。
- attention全体を一つのcheckpoint領域にする構成を維持し、backward時に中間値を再計算する。
- step/predictのキャッシュ形式、モデルの重み名・形状、保存configは変更しない。

最初の試作では全ての端数を別処理にしていたが、batch1で39.35→50.58 msと遅くなった。
上記の上限付きパディングで小さな行列積の呼び出し回数を抑えた。
試作の[計測記録](experiments/logkv-grouped-forward-20261029/block-b1.json)とソースは保持している。

## 検証

- CPU全体：**489 passed / 21 skipped**。
- GPUを含むforward検証：**91 passed**。
  独立FP64参照、出力・入力勾配・重み勾配、CUDA FP32/BF16/autocastを含む。
- 新規42件では階層ごとの統計量とQ/K/V/レベルバイアスの勾配をgather版と比較。
  C=2/3/4、複数階層、全slot無効、端数、境界整列、空入力、再計算on/offを確認。
  forward/backwardがgather版に戻らないことも確認した。
- 固定減衰・固定0それぞれで2GPU統合テストが通過。
  loss/勾配/global clip、保存・export・推論、層分割変更、同期終了、
  中断再開の重み・optimizer・RNGのbit一致を確認。
  再開のbit一致は新実装内の連続実行との比較であり、旧実装との一致ではない。

ゲート付きMamba2のCPU forwardは従来step経由だったため、gate=0でのLogKVとの
厳密一致テストで最大約3e-8の差が出た。pastなし・cache不要のCPU呼び出しも
LogKVと同じ専用forwardへ通すことで既存の厳密一致テストを維持した。
Mamba2のCPU特徴抽出、CUDAの公式backbone、キャッシュ経路はそれぞれ従来の方式を使う。
追加のMamba2 CUDAテストはPython 3.12開発ヘッダー`Python.h`不足により
Tritonのコンパイルで停止したため、今回の環境では未確認。
LogKV自体のCUDAテストと2GPU統合テストは成功している。

[検証記録](experiments/logkv-grouped-forward-20261029/review.json)。
大きな統合テスト成果物と計測時ソースは
`/mnt/raid0/RecursiveCompressor/experiments/logkv-grouped-forward-20261029-hcmGcM/`に保存した。
本学習・既存runの再開は行っていない。

## 1. の実装との速度比較

比較対象は`9610511`相当の状態保存なし＋gather方式。
同じモデル・重み・合成入力に対して`_attend_full_contexts`だけをgather方式に切り替える。
RTX 3090、FP32 master＋BF16 autocast、C4、Conv4、phaseなし、gate/self slotあり、固定減衰。
入力長2047、3回warmup、交互順7組×各3回の平均時間の中央値。

計測対象は**forward＋backward＋勾配クリア**。
2GPUではtrainerと同じPipelineEngine/1F1Bの入力転送・通信を含む。
optimizer更新・生成・checkpoint I/Oを含む実訓練全体の速度ではない。

| 対象 | 1. の実装 ms | K/V共有 ms | 時間短縮 | ピーク割当MiB（従来→新） |
|---|---:|---:|---:|---:|
| 1 Block、d1024/FF3072、batch1 | 41.68 | 37.40 | 10.3% | 531.3 → 393.6 |
| 1 Block、d1024/FF3072、batch4 | 84.52 | 47.63 | 43.7% | 1658.2 → 1204.6 |
| 2GPU、4層、d512/FF1536、batch4 | 289.74 | 270.47 | 6.6% | rank0: 550.7 → 481.9、rank1: 383.6 → 314.5 |

パイプラインは2層ずつ、vocab1024、4 microbatches、accum1、先頭1/3のラベルを除外。
batch4の単一Blockでは約1.77倍、パイプラインでは約1.07倍の処理速度だった。
結果はこの小規模構成の測定値であり、大規模モデルへの同率の改善は保証しない。

BF16の丸め順が変わるため、旧経路とのbit一致は保証しない。
単一Blockのlossは一致し、勾配の最大絶対差は約7.63e-6。
パイプラインloss差は約4.77e-7、勾配の最大絶対差はrank0で0.000385、rank1で0.0000191、
勾配全体の相対L2差は約0.759% / 0.336%だった。

[batch1](experiments/logkv-grouped-forward-20261029/block-b1-bounded-padding.json) /
[batch4](experiments/logkv-grouped-forward-20261029/block-b4.json) /
[2GPU](experiments/logkv-grouped-forward-20261029/pipeline-2gpu.json)。
生データには全反復、rankごとのピーク割当、勾配差、環境、計測時ソースhashを収録した。

## 再現コマンド

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/logkv/test_logkv_grouped_forward.py tests/logkv/test_logkv_forward.py -q

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m benchmarks.benchmark_logkv_forward --target block --baseline stateless \
  --dim 1024 --ff 3072 --batch-size 1 --output /tmp/grouped-b1.json
# batch4は --batch-size 4 と別のoutputを指定する。

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  --module benchmarks.benchmark_logkv_forward --target pipeline --baseline stateless \
  --dim 512 --ff 1536 --layers 4 --vocab 1024 --batch-size 4 --microbatches 4 \
  --output /tmp/grouped-pipeline.json
```
