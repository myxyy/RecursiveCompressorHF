[English](README_en.md) | 日本語

# RecursiveCompressor

再帰的な圧縮機構を持つ独自アーキテクチャによる言語モデルの実装です。

## アーキテクチャ

RecursiveCompressorは、入力シーケンスをチャンクに分割し、各チャンクをcausal attentionで処理した後、チャンク間の情報伝達を再帰的な圧縮・展開によって実現するモジュールです。

言語モデル（RecursiveCompressorLM）は以下の構造を持ちます:

```
Embedding → RecursiveCompressor × num_layers → LayerNorm → Linear
```

HuggingFaceの `PreTrainedModel` を継承しており、`save_pretrained` / `from_pretrained` / `push_to_hub` に対応しています。

## セットアップ

```bash
uv sync
cp .env.example .env
# .env の DATA_DIR を編集（データセット・チェックポイントの保存先）
```

## 使い方

### 学習

```bash
# 単一GPU
uv run python train.py

# マルチGPU（例: 6GPU）
uv run torchrun --nproc_per_node=6 train.py
```

学習データはHuggingFaceから自動ダウンロードされ、トークナイズ済みキャッシュ（numpy memmap）が `$DATA_DIR/hf_cache/mmap/` に保存されます。2回目以降はキャッシュから高速にロードされます。

#### 学習中の制御

```bash
echo "pause"         > control.cmd   # 一時停止
echo "resume"        > control.cmd   # 再開
echo "save_and_exit" > control.cmd   # 保存して終了
```

チェックポイントは `$DATA_DIR/checkpoints/` に保存され、再起動時に自動復帰します。

### テキスト生成

```bash
uv run python predict.py "吾輩は猫である。" --model-dir ./data/final_model \
    --context-length 256 --temperature 0.8
```

| オプション | 説明 | デフォルト |
|---|---|---|
| `prompt` | 入力テキスト（必須） | - |
| `--model-dir` | モデルディレクトリ（必須） | - |
| `--context-length` | 最大生成トークン長 | 1024 |
| `--temperature` | サンプリング温度 | 1.0 |

### テスト

```bash
uv run pytest test_lm.py -v
```

## ファイル構成

| ファイル | 説明 |
|---|---|
| `recursive_compressor.py` | RecursiveCompressorモジュール（`step`/`forward`/`predict`） |
| `recursive_compressor_lm.py` | 言語モデル（PreTrainedModel継承） |
| `configuration_recursive_compressor.py` | モデル設定（PretrainedConfig継承） |
| `dataset.py` | HFデータセット読み込み・トークナイズ・memmapキャッシュ |
| `train.py` | DDP学習スクリプト（チェックポイント・制御コマンド対応） |
| `predict.py` | テキスト生成（`from_pretrained`でモデルロード） |
| `test_lm.py` | テスト |
| `.env.example` | 環境設定例 |

## 学習データセット

| データセット | 種類 |
|---|---|
| `wikimedia/wikipedia` (ja, en) | 文章 |
| `JeanKaddour/minipile` | 文章 |
| `shi3z/ja_conv_wikipedia_llama2pro8b_30k` | 対話 |
| `shi3z/ja_conv_wikipedia_orion14B_100K` | 対話 |
| `HuggingFaceH4/ultrachat_200k` | 対話 |

文章データは `[DOC]` マーカー付き、対話データは `[QUERY]`/`[ANSWER]` マーカー付きでフォーマットされます。

## モデルパラメータ

| パラメータ | 値 |
|---|---|
| d_model | 1024 |
| num_heads | 8 |
| d_ff | 2048 |
| chunk_size | 8 |
| compress_size | 4 |
| num_layers | 8 |
| context_length | 4096 |
| optimizer | RAdamScheduleFree |
| learning_rate | 3e-4 |
