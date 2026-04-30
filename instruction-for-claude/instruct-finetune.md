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
