# LogKVの圧縮Query・Keyオプション（2026-11-20）

通常のDDP trainerとpipeline trainerで、次の3引数を組み合わせて訓練できる。

```text
--compression-query next --compression-key mean --level-amplify
```

| 引数 | 選択肢 | 新規訓練の既定 |
|---|---|---|
| `--compression-query` | `last`: 区間末尾、`next`: 区間直後のQuery | `last` |
| `--compression-key` | `attention`: Attention重み付き平均、`mean`: 算術平均 | `attention` |
| `--level-amplify` | 読み出しの固定バイアスを `+i log C` にする | 無効（`-i log C`） |

QueryとKeyの選択は独立で、4通りを比較できる。Valueはどの方式でも圧縮Queryによる
softmax重み付き平均。追加の学習パラメータはなく、既存の重み名・形状を維持する。
`--level-amplify`は既存オプションで、`--no-level-bias` / `--learnable-decay` と排他。

## 訓練コマンド

`.env`の`DATA_DIR`にデータとcheckpointの保存先を設定する。以下は2GPU、1 epochの例。
batchと層分割はモデルサイズ・VRAMに合わせて調整する。

```bash
# パイプライン並列（標準16層を8層ずつに分割）
uv run torchrun --standalone --nproc_per_node=2 --module training.train_logkv_pipeline \
  --run-name pipeline-next-mean-amplify \
  --batch-size 4 --n-microbatches 4 --grad-accum 1 \
  --stage-layer-split 8,8 --num-epochs 1 \
  --compression-query next --compression-key mean --level-amplify

# DDP（モデル全体が各GPUに収まる場合）
uv run torchrun --standalone --nproc_per_node=2 --module training.train_logkv \
  --run-name ddp-next-mean-amplify --batch-size 2 --num-epochs 1 \
  --compression-query next --compression-key mean --level-amplify
```

既存の固定減衰モデルから重みを引き継いで方式を変更する場合は、新しいrun名と
pipelineの`--start-checkpoint`を使う。

```bash
uv run torchrun --standalone --nproc_per_node=2 --module training.train_logkv_pipeline \
  --run-name pipeline-next-mean-warm \
  --start-checkpoint /path/to/standard-checkpoint/model \
  --batch-size 4 --n-microbatches 4 --grad-accum 1 --num-epochs 1 \
  --compression-query next --compression-key mean --level-amplify
```

checkpoint由来の層数を使い、層分割は省略時に均等配置する。
重みのみを読み込み、optimizer・step・データ位置は最初から開始する。

- 圧縮引数の省略時は保存済み設定を保持する。`last` / `attention`への逆切り替えも明示指定で可能。
- `--level-amplify`で固定減衰または固定0から固定増幅へ切り替えられる。
  学習可能減衰からの切り替えは、パラメータの暗黙の削除を防ぐため拒否する。
- `--resume`は保存時の方式を保持する。圧縮・増幅引数を付け直す必要はなく、
  異なる方式の明示指定は拒否する。同じ方式の明示指定は可能。
- 旧configに圧縮項目がない場合は`last` / `attention`として読み込む。
  新設定は`config.json`に保存され、通常の推論CLIやHFモデル読み込みでも復元される。

## 圧縮タイミングと計算

`next`ではC=4の区間0〜3を`q4`で圧縮する。位置16では、12〜15の圧縮と、
4個の子要約をまとめる0〜15の圧縮をともに`q16`で行う。
上位でQueryをさらにずらすわけではない。

- `forward`: `[a,b)`の圧縮は元系列の`q[b]`がある場合だけ構築。
  レベルごとにまとめて圧縮し、既存のgrouped QK/PV読み出しを維持する。
- `step`: segment末尾の完成したraw chunkを保持し、次segmentに境界Queryが到着すると圧縮する。
  token単位のPythonループは追加していない。
- `predict`: 境界Queryによる圧縮・上位への繰り上げ、読み出し、現在KVの保存の順に実行する。
  `step`は呼ばない。

圧縮要約は境界位置以降だけから参照され、未来情報は漏れない。
hiddenの外形は`(levels, offset)`のまま。`next`ではlevel 0に最大C個、
上位には最大C−1個を保持する。推論状態は引き続き`O(C log n · d)`。
方式の異なる実行途中のhiddenは共用せず、切り替え時は`hidden=None`から始める。

## 一次近似の適用範囲

同じQueryで圧縮・読み出しをする単段の境界では、区間の正確な重みの総量
`sum(exp(s_j))`を、平均Keyと増幅により`C * exp(mean(s_j))`で近似する。
区間内のスコア差が小さいと一次まで一致し、誤差は二次から生じる。
後続QueryではValueが以前のQuery向けに固定され、多段圧縮も子内部の分配を変更できないため、
元の全トークンAttentionとの一次一致は一般には保証しない。

`--kv-norm` / `--v-norm-only`との併用は実装上可能だが、圧縮後の再正規化により
上記の算術平均・Value平均の条件を変える。一次近似を検証する構成では両方無効、
self slot有効とする（いずれも新規訓練の既定）。
平均Keyのスコアが小さいだけでは近似精度は保証されず、区間内のばらつきにも依存する。
モデル品質・訓練速度の改善は、この実装検証では評価していない。

## 検証

2026-11-20の検証結果：

- 全CPU回帰：**561 passed / 21 skipped**。GPU専用などのケースはCPU実行でskip。
- 新規72件：独立参照による圧縮検証53件、設定・checkpoint検証19件。
- 2GPU統合：`next + mean + level_amplify`、小型3層モデル・合成データで成功。
  全モデルとpipelineのloss/勾配/global clipping、BF16訓練、分散サンプル生成、
  epoch境界を越える中断再開、export、層分割を変えたwarm-start、同期終了、
  `predict_stream`経由の読み込み・生成を確認。
  圧縮・増幅引数を省略した再開でも、連続実行と重み・optimizer・乱数状態がbit一致した。
- 変更前のHEADの実装と既定方式をFP64/FP32/BF16で直接比較し、
  forward/step/predictがbit一致。新方式の1025トークン・C=2/3/4/8の追加確認でも、
  不規則なsegment分割とのFP64誤差は1e-12以内、保持tensorが余分なstorageを参照しないことを確認。

[機械可読の検証記録](experiments/logkv-compression-options-20261120/review.json)。
小規模GPU検証のcheckpoint・ログ・CPU結果XMLは
`/mnt/raid0/RecursiveCompressor/experiments/logkv-compression-20261120-I7tGZ9/`に保存。
本学習・既存runの再開は行っていない。

`tests/logkv/test_logkv_compression.py`は圧縮器・階層構築の実装を呼ばない
独立した区間参照モデルで、4方式、C=2/3/4、forward/分割step/predict、勾配、
4/16/64境界、未来情報の遮断、状態非破壊とstorage量、grouped経路を確認する。
設定・HF保存復元・warm-start・resumeは`test_logkv_compression_config.py`で検証する。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  uv run --no-sync python -m pytest tests/logkv -q

# 毎回新しい保存先を指定。合成データ・小さなモデルの2GPU統合検証。
CUDA_VISIBLE_DEVICES=0,1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync python -m torch.distributed.run --standalone --nproc_per_node=2 \
  --module tests.logkv.test_logkv_pipeline --distributed-smoke /path/to/new/smoke-dir \
  --compression-query next --compression-key mean --level-amplify
```
