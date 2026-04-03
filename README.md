[English](README_en.md) | 日本語

# RecursiveCompressor

再帰的な圧縮機構を持つ独自アーキテクチャによる言語モデルの実装です。

## アーキテクチャ

RecursiveCompressorは、入力シーケンスをチャンクに分割し、各チャンクをcausal attentionで処理した後、チャンク間の情報伝達を再帰的な圧縮・展開によって実現するモジュールです。

言語モデル（RecursiveCompressorLM）は以下の構造を持ちます:

```
Embedding → RecursiveCompressor × num_layers → LayerNorm → Linear
```

## セットアップ

```bash
uv sync
```

## 使い方

### 学習

`text/` ディレクトリに学習用の `.txt` ファイルを配置し、以下を実行します。

```bash
uv run python train.py
```

トークナイザには `elyza/ELYZA-japanese-Llama-2-7b-fast` を使用します。

### テキスト生成

```bash
# 高速版（隠れ状態を引き継いで1トークンずつ生成）
uv run python predict_fast.py "吾輩は猫である。" --context-length 256 --temperature 0.8

# ナイーブ版（毎回全シーケンスをforward）
uv run python predict_naive.py "吾輩は猫である。" --context-length 256 --temperature 0.8
```

| オプション | 説明 | デフォルト |
|---|---|---|
| `prompt` | 入力テキスト（必須） | - |
| `--context-length` | 最大生成トークン長 | 1024 |
| `--temperature` | サンプリング温度 | 1.0 |
| `--weights` | 重みファイルのパス | `recursive_compressor_lm.pth` |

### テスト

```bash
uv run pytest test_lm.py -v
```

## ファイル構成

| ファイル | 説明 |
|---|---|
| `recursive_compressor.py` | RecursiveCompressorモジュール |
| `recursive_compressor_lm.py` | 言語モデル（RecursiveCompressorLM） |
| `dataset.py` | テキストデータセット（TextDataset） |
| `train.py` | 学習スクリプト |
| `predict_fast.py` | 高速テキスト生成（隠れ状態再利用） |
| `predict_naive.py` | ナイーブテキスト生成（毎回full forward） |
| `test_lm.py` | テスト |

## デフォルトハイパーパラメータ

| パラメータ | 値 |
|---|---|
| d_model | 512 |
| num_heads | 8 |
| d_ff | 2048 |
| chunk_size | 8 |
| compress_size | 4 |
| num_layers | 4 |
| context_length | 2048 |
| batch_size | 4 |
| learning_rate | 3e-4 |
