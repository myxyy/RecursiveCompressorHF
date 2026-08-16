## やること
会話データセットによるファインチューニング

## 1. データセットの用意
`prepare_all_datasets()`メソッドにデータセット種別を指定する引数`dataset_type`を追加する。`dataset_type`に指定できる文字列は次の2つ

* `pretrain` - 現在の`prepare_all_datasets()`で用意されているデータセット
* `instruct` - `shi3z/ja_conv_wikipedia_llama2pro8b_30k`と`shi3z/ja_conv_wikipedia_orion14B_100K`だけで構成される

## 2. 訓練処理
train_pipeline.pyの起動時に`--dataset-type`引数を`pretrain`（デフォルト）、`instruct`から指定して選択できるようにする。また、DATA_DIR以下checkpointのフォルダ名は`chekpoints_pipeline/checkpoint-pretrain-13000`や`checkpoints_pipeline/checkpoint-instruct-23000`のようにデータセット種別を含めるようにし、起動時のデータセット種別に応じて再開できるようにする。

train_pipeline.pyでは起動時引数`--start-checkpoint`としてチェックポイントディレクトリを指定できるようにする。これにより事前学習のチェックポイントから会話ファインチューニングができるようになる。  
尚、チェックポイントディレクトリを指定した場合でもデータセット種別に対応したチェックポイントディレクトリが存在する場合はそのチェックポイントから再開するようにする。（チェックポイントディレクトリが存在する場合に`--start-checkpoint`が指定された場合は何か警告を出したい）

## 実装結果

### `dataset.py`
- `DATASET_TYPES = ("pretrain", "instruct")` を公開
- `prepare_all_datasets(..., dataset_type="pretrain")` 引数追加
  - `pretrain`: wiki_ja + cc100_ja + shi3z 2種
  - `instruct`: shi3z 2種のみ

### `train_pipeline.py`
- CLI:
  - `--dataset-type {pretrain,instruct}` (デフォルト `pretrain`)
  - `--start-checkpoint /path/to/checkpoint` (オプション)
- チェックポイント命名: `checkpoint-{dataset_type}-{step}` 例: `checkpoint-pretrain-13000`, `checkpoint-instruct-23000`
- 再開ロジック:
  1. 同じ`dataset_type`の既存チェックポイントを優先して再開
  2. なければ`--start-checkpoint`指定があれば、そのチェックポイントの`full_model.pt`から**モデル重みのみ**ロード（オプティマイザはスクラッチから開始）
  3. どちらもなければ完全にスクラッチ開始
  4. **警告**: 既存チェックポイントがあるのに`--start-checkpoint`が指定された場合、その指定は無視されて警告ログが出力される
- 最終モデル保存先: `${DATA_DIR}/final_model_{dataset_type}/`

### 使用例
```bash
# 事前学習
uv run torchrun --nproc_per_node=6 train_pipeline.py --dataset-type pretrain

# 事前学習チェックポイントから会話ファインチューニング
uv run torchrun --nproc_per_node=6 train_pipeline.py \
  --dataset-type instruct \
  --start-checkpoint /mnt/raid0/RecursiveCompressor/checkpoints_pipeline/checkpoint-pretrain-100000

# instructチェックポイントから再開（--start-checkpoint不要）
uv run torchrun --nproc_per_node=6 train_pipeline.py --dataset-type instruct
```
