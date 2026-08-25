# HuggingFace活用
RecursiveCompressorをHuggingFaceのインフラを活用して学習し、モデルのアップロードの準備もする

## モデル
https://huggingface.co/docs/hub/models-uploading
に基づいて`RecursiveCompressorLM`を`push_to_hub`や`from_pretrained`等のメソッドが使えるような形にする
RecursiveCompressorを試したい人がレポジトリクローンして`from_pretrained`でHuggingFaceからモデル落として使えるのが目標

### モデルパラメータ
ひとまず
*  d_model = 1024
*  num_heads = 8
*  d_ff = 2048
*  chunk_size = 8
*  compress_size = 4
*  num_layers = 8

として学習の様子を見て調整する  
現状trainとpredictにパラメータ記述が分散してしまっているがどこかにまとめたい  
そのあたりもHuggingFaceのインフラで吸収できる？要調査  

## データセット
通常の文章データセットとマルチターン対話データセットを混ぜて使用する  
基本的に全ての文字列はモデルに読ませる際にはBOSトークン`<s>`を文頭と文末につけるようにする  
ただし学習コンテキスト長を超える場合は文末のBOSトークンは除くこと（生成コンテキスト長は事実上制限がないので続きを生成させたい意図がある）  
各データセットの格納の形式は適宜調査すること  

### 文章データセット
先頭に`<s>[DOC]`を付けて学習させる  
例：「本日は晴天なり」は`<s>[DOC]本日は晴天なり<s>`の形で読み込まれる  

使用データセットは以下
* `wikimedia/wikipedia`
  - `20231101.en`と`20231101.ja`のサブセットを使用する
* `JeanKaddour/minipile`

### 対話データセット
マルチターン対話は  
`<s>[QUERY]質問文1[ANSWER]回答文1<s>[QUERY]質問文2[ANSWER]回答文2<s>...`  
の形で学習される  
例：

Q: 猫の足は何本ですか  
A: 4本です  
Q: 足には爪が生えていますか  
A: はい  

という対話の場合  
`<s>[QUERY]猫の足は何本ですか[ANSWER]4本です<s>[QUERY]足には爪が生えていますか[ANSWER]はい<s>`  
の形で読み込まれる  

使用データセットは以下
* `shi3z/ja_conv_wikipedia_llama2pro8b_30k`
* `shi3z/ja_conv_wikipedia_orion14B_100K`
* `HuggingFaceH4/ultrachat_200k`


## 学習方法
GPU並列で学習を行う

### パディング
各データはコンテキスト長に満たない分はPADトークンで埋め、PADトークンに対応する予測からは勾配が流れないようにする。

### 並列学習
`torchrun`コマンドでGPU並列DDP学習ができるようにする  

### 学習パラメータ
* コンテキスト長 4096
* バッチサイズ GPU数に応じる、6GPUだったら6
* エポック数 1
* 学習率 3e-4 様子を見て調整する

### オプティマイザ
[schedule_free](https://github.com/facebookresearch/schedule_free)レポジトリで提供されるRAdamScheduleFreeを使用する
バリデーションデータはひとまずデータセットの0.1%を分割して用いる

## 長期学習の運用方法

### バックアップ
一定ステップ毎にチェックポイントを作成し、最新2件を保存しておく
再開時は最新のチェックポイントから再開する

### 操作コマンド
学習中のスレッドに対して外部から一時停止、再開、保存して終了などのコマンドを送信できるようにしたい  
計算資源が家にあるVRAM24GBのRTX3090が6枚で、訓練中は電子レンジ等が使えないので一時停止と再開は必要  

## データの置き場所
データセットと重みは`/mnt/raid0`以下に適当なディレクトリ構造を作成して保存したい  
ただしこの文書を除いて`/mnt/raid0`に保存されるという情報はGitの追跡からは隠蔽し、レポジトリをクローンした人が保存先ディレクトリを自由に設定できるようにしたい  
.envファイル等で設定するのが適切だろうか  

## 備考
やりたいことが色々あるので計画立てて実装してください  
進捗は適宜コミットを積んでください  
このドキュメントは適宜追記お願いします  
以上の記述において不適切と思われる点や不明点あったら質問ください  

---

## 実装結果

### ファイル構成
| ファイル | 役割 |
|---|---|
| `configuration_recursive_compressor.py` | `RecursiveCompressorConfig` (PretrainedConfig) — モデルパラメータを一元管理 |
| `recursive_compressor_lm.py` | `RecursiveCompressorLM` (PreTrainedModel) — `save_pretrained`/`from_pretrained`/`push_to_hub`対応 |
| `dataset.py` | HFデータセット読み込み、フォーマット(`[DOC]`/`[QUERY]`/`[ANSWER]`)、トークナイズ、PADマスク |
| `train.py` | DDP学習スクリプト — `torchrun`対応、RAdamScheduleFree、チェックポイント、制御コマンド |
| `predict.py` | 推論スクリプト — `from_pretrained`でモデルロード、`step`でプロンプト一括処理 |
| `.env.example` | データディレクトリ設定例 (`DATA_DIR=./data`) |

### パラメータの一元管理
`RecursiveCompressorConfig` に全モデルパラメータを集約。`config.json`として保存され、`from_pretrained`で自動復元される。trainとpredictのパラメータ分散は解消。

### データセットフォーマット
- 文章: `<s>[DOC]text<s>` (コンテキスト長超過時は末尾`<s>`省略)
- 対話: `<s>[QUERY]q[ANSWER]a<s>[QUERY]q[ANSWER]a<s>`
- PAD部分のlabelsは`-100`で勾配が流れない

### 制御コマンド
`control.cmd`ファイルに書き込むことで学習を制御:
```bash
echo "pause"         > control.cmd   # 一時停止
echo "resume"        > control.cmd   # 再開
echo "save_and_exit" > control.cmd   # 保存して終了
```

### データの置き場所
`.env`ファイルで`DATA_DIR`を設定（例: `DATA_DIR=/mnt/raid0/recursive_compressor`）。
`.env`は`.gitignore`に追加済み、`.env.example`をリポジトリに含む。

### 実行方法
```bash
# .envを作成
echo "DATA_DIR=/mnt/raid0/recursive_compressor" > .env

# 6GPU学習
uv run torchrun --nproc_per_node=6 train.py

# 推論
uv run python predict.py "吾輩は猫である。" --model-dir /mnt/raid0/recursive_compressor/final_model
```