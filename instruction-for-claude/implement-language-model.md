# RecursiveCompressorLMの実装
## やること
RecursiveCompressorモジュールを言語モデルにして動作確認する

## モデル
通常のTransformerDecoderモデルと同様に、RecursiveCompressorモジュールを複数段積み重ねたものを埋め込み層とLayerNorm+線形層で挟んだもので次単語予測を行う。  

## データセット
textディレクトリにある.txtファイルをすべて結合し、適当な長さに切断したものをデータセットとする。  
切断長さはトークナイズ後のコンテキスト長（1024程度を考えている）が一定となるのが望ましいが、トークナイズの都合により適切な形になるようにしたい。  

## トークナイザ
日本語が使えるものならなんでもよいが、テスト用の計算資源はVRAM24GBのRTX3090のためあまり大きいものは避けたい
参考として別で動かしていたプロジェクトではHuggingfaceの`elyza/ELYZA-japanese-Llama-2-7b-fast`のトークナイザを使用していた

## 実装構成

### ファイル構成
- `recursive_compressor_lm.py` — 言語モデル本体 (`RecursiveCompressorLM`)
- `dataset.py` — テキストデータセット (`TextDataset`)
- `train.py` — 学習スクリプト
- `test_lm.py` — テスト

### RecursiveCompressorLM
`Embedding` → `RecursiveCompressor` × num_layers → `LayerNorm` → `Linear`（vocab射影、bias無し）

コンストラクタ引数:
- `vocab_size` — トークナイザの語彙数
- `d_model` — 埋め込み次元
- `num_heads` — アテンションヘッド数
- `d_ff` — FFN中間次元
- `chunk_size` — RecursiveCompressorのチャンクサイズ
- `compress_size` — RecursiveCompressorの圧縮サイズ
- `num_layers` — RecursiveCompressorの積み重ね数

### TextDataset
`text/` 配下の全 `.txt` ファイルを結合し、指定トークナイザでトークナイズ後、`context_length` ごとに切断。
各サンプルは `(input_ids, target_ids)` のペアで、target は input を1トークンずらしたもの。

### トークナイザ
`elyza/ELYZA-japanese-Llama-2-7b-fast` を使用（`transformers.AutoTokenizer`経由）。

### 学習ハイパーパラメータ（デフォルト）
| パラメータ | 値 |
|---|---|
| context_length | 1024 |
| d_model | 512 |
| num_heads | 8 |
| d_ff | 2048 |
| chunk_size | 16 |
| compress_size | 8 |
| num_layers | 4 |
| batch_size | 4 |
| num_epochs | 10 |
| learning_rate | 3e-4 |
| optimizer | AdamW |
| gradient clipping | 1.0 |

### 実行方法
```bash
uv run python train.py
```

### テスト
```bash
uv run pytest test_lm.py -v
```
