# LogKVのパイプライン並列訓練

`train_logkv_pipeline.py`は`train_pipeline.py`の`Schedule1F1B`を参考にした、LogKV用の
パイプライン並列trainer。1 GPUに1ステージを配置する。
通常の`train_logkv.py`とモデル構造・設定オプション・Muon/AdamWの振り分け・linear warmupを共有する。
既定はConv幅4、gate/self slotあり、phaseなし、固定減衰。

## 訓練と生成

`.env`の`DATA_DIR`を設定してから実行する。以下は6GPU、標準16層の例。

```bash
uv run torchrun --standalone --nproc_per_node=6 train_logkv_pipeline.py \
  --run-name pipeline-base \
  --batch-size 12 --n-microbatches 12 --grad-accum 2 \
  --num-layers 16 --stage-layer-split 2,2,3,3,3,3 \
  --max-steps 5000
```

この例では各microbatchは1サンプル、有効バッチは24サンプル。
`--batch-size`はパイプライン全体で共有する1回のscheduleのバッチで、GPU数は掛けない。
有効バッチは`batch-size × grad-accum`、microbatchサイズは`batch-size / n-microbatches`。
`batch-size`は`n-microbatches`で割り切れ、`n-microbatches >= GPU数`が必要。
各ステージは1層以上持つ。

`--stage-layer-split`省略時は均等分割（余りは前段）。1F1Bでは前段ほど同時に保持する
microbatchのactivationが多いため、メモリが偏る場合は前段の層数を減らして調整できる。
上記の層配分は指定方法の例であり、大規模構成でのVRAM使用量・速度の測定値ではない。
重みとoptimizerはfp32、演算は既定bf16 autocast。`--precision fp32`も指定可能。

保存先：

```text
$DATA_DIR/checkpoints_logkv_pipeline/pipeline-base/checkpoint-5000/
  model/
    config.json
    model.safetensors          # 大きい場合はHF形式の複数shard
    generation_config.json
    tokenizer_config.json     # tokenizer本体・special token設定も同じ場所
    ...
  stage_0.pt ...               # 各ステージの重み
  optimizer_0.pt ...           # optimizer・CPU/CUDA/Python乱数状態
  trainer_state.json          # step、epoch内位置、EMA、分割とデータ条件
```

毎回のcheckpointに推論可能な`model/`が含まれる。次の指定で既存のストリーム生成が使える。
`DATA_DIR`がshell変数にも設定されている場合の例：

```bash
uv run python predict_stream.py \
  --model-dir "$DATA_DIR/checkpoints_logkv_pipeline/pipeline-base/checkpoint-5000/model" \
  --device 0 --precision bf16
```

`predict_stream.py`の変更は不要。`config.json`の`model_type=logkv`によって通常のLogKVLMとして読む。
訓練を分散できても、この生成CLIは統合したモデル全体を指定デバイスにロードする。
大きいモデルの推論にはその分のVRAM、または`--device cpu --precision fp32`を使うメモリが必要。

## 再開・重みだけの引継ぎ

同じrun-name、層分割、batch/accum/microbatch、context、seed、precision、dataset条件を指定して再開する。
`--max-steps`は再開後の追加数ではなく絶対ステップ数。

```bash
uv run torchrun --standalone --nproc_per_node=6 train_logkv_pipeline.py \
  --run-name pipeline-base \
  --batch-size 12 --n-microbatches 12 --grad-accum 2 \
  --stage-layer-split 2,2,3,3,3,3 \
  --resume latest --max-steps 10000
```

checkpointには次に消費するepochとepoch内batch位置を保存する。epoch境界でも消費済みサンプルを
再読せず、全体のoptimizer step数からepoch内位置を推測しない。
モデル構造はcheckpointのconfigを採用するため、resume時のモデル構造CLI引数は使わない。
`--lr`と`--warmup`は指定値を使い、保存したstepからLRを計算する。

GPU数や層配分を変える場合、または通常のDDP版LogKVから引き継ぐ場合は、
新しいrun-nameで`--start-checkpoint /path/to/checkpoint/model`を指定する。
`model/`を含むcheckpoint親ディレクトリの指定にも対応する。
この場合は重みのみをロードし、step・optimizer・データ位置は最初から開始する。
HF safetensors形式（shard分割を含む）を読む。
`--resume`と`--start-checkpoint`は同時に指定できない。

新規訓練で既存checkpointのあるrunを使うこと、同じrunへの二重起動、同じ出力runの古い
checkpointから後続checkpointを上書きすることは拒否する。分岐させる場合は新しいrun-nameを使う。

## 制御と記録

`just pause`、`just resume`、`just save-and-exit`は既存trainerと同様に使える。
複数runを起動する場合は`--control-file /path/to/that-run.cmd`で制御ファイルを分けられる。
`pause`中にも`save_and_exit`を受け付ける。

`--sample-interval`の既定は1000、0で無効。日本語の3プロンプトからのサンプルを
同じパイプラインで生成するため、サンプル生成時にも全モデルを1 GPUへ集めない。
既定80トークン、`--sample-max-new-tokens`で変更可能。samplingの乱数は訓練から分離し、
結果はrun内の`samples.log`へ記録する。
TensorBoardは`$DATA_DIR/tensorboard/logkv-pipeline/{run-name}`。

checkpointは全rankの保存とCPUでのHF形式への統合が完了してから公開する。
途中で停止した`.tmp`は`--resume latest`の候補に入れない。
`--max-checkpoints`の既定は2で、公開後に古いcheckpointを削除する。
保存時はrank 0のCPUメモリにモデル全体の重みを集める。optimizerファイルは別にし、
統合時に全optimizerを同時に読み込まない。出力とデータキャッシュは全rankから見える共有filesystemが必要。

## 計算と検証

ステージ間には`(batch, sequence, d_model)`の通常の隠れ状態を渡す。
各rankは担当層だけを構築する。保存する層名は全モデルのグローバルな番号を保つ。
勾配クリッピングは全ステージの二乗ノルムを合算し、全モデル共通の係数で行う。

lossは蓄積する全batchの**有効ラベル総数**で正規化する。
microbatchごとの平均を単純平均しないため、instructの応答長やPAD数が違っても
全バッチをまとめたcross entropyと整合する。全ラベルが`-100`のmicrobatchでもlossと勾配は有限の0になる。
このtoken単位の重み付けは、DDP版のrank/batchごとの平均lossとは一般に同じではない。
新規初期化の乱数はseed+rankを使い、同じseedだけでDDP版と同一初期値になるとは保証しない。

2026-09-16、以下の回帰テストは**230件すべて通過**した。
[統合テストの確認記録](experiments/logkv-pipeline-20260916/review.json)。

検証コマンド：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest test_logkv.py test_logkv_lm.py test_logkv_conv.py \
  test_logkv_predict.py test_logkv_pipeline.py -q

# 空の保存先を指定。人工データ・小さいモデルだけの統合テスト。
CUDA_VISIBLE_DEVICES=0,1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  test_logkv_pipeline.py --distributed-smoke /mnt/raid0/RecursiveCompressor/pipeline-smoke-new
```

2GPUの統合テストでは、不均一なマスクを含む全モデルとのloss/勾配比較、global clipping、
bf16での訓練・分散サンプル生成、epoch途中から再開して境界を越えたときの重み・optimizer・乱数のbit一致、
checkpointローテーション、層配分変更時の重み引継ぎ、save_and_exit、
保存したモデルの`predict_stream.py`経由の読込・生成を確認した。
大規模LLMの本学習・スループット評価はこの実装確認には含めていない。
