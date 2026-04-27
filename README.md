[English](README_en.md) | 日本語

# RecursiveCompressor

再帰的な圧縮機構を持つ独自アーキテクチャによる言語モデルの実装です。

## アーキテクチャ

RecursiveCompressorは、入力シーケンスをチャンクに分割し、各チャンクをcausal attention（GatedAttention）で処理した後、チャンク間の情報伝達を再帰的な圧縮・展開によって実現するモジュールです。

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

### 学習（パイプライン並列）

```bash
uv run torchrun --nproc_per_node=6 train_pipeline.py
```

モデルがGPU間に分割されるパイプライン並列方式で学習します。混合精度（fp32マスター重み + bfloat16 autocast）で実行され、Muon（隠れ層の2D重み）+ AdamW（embedding/head/bias/LayerNorm）の2段オプティマイザを使用します。

学習データはHuggingFaceから自動ダウンロードされ、トークナイズ済みキャッシュ（numpy memmap）が `$DATA_DIR/hf_cache/mmap/` に保存されます。2回目以降はキャッシュから高速にロードされます。

#### 学習中の制御

```bash
echo "pause"         > control.cmd   # 一時停止
echo "resume"        > control.cmd   # 再開
echo "save_and_exit" > control.cmd   # 保存して終了
```

チェックポイントは `$DATA_DIR/checkpoints_pipeline/` に保存され、再起動時に自動復帰します。

### テキスト生成

```bash
# 1回生成
uv run python predict.py "吾輩は猫である。" --model-dir /path/to/checkpoint \
    --context-length 256 --temperature 0.8 --top-p 0.9

# 対話的にストリーム生成
uv run python predict_stream.py --model-dir /path/to/checkpoint \
    --context-length 1024 --temperature 0.8 --top-p 0.9
```

| オプション | 説明 | デフォルト |
|---|---|---|
| `prompt` | 入力テキスト（predict.pyのみ必須） | - |
| `--model-dir` | モデルディレクトリ（必須） | - |
| `--context-length` | 最大生成トークン長 | 1024 |
| `--temperature` | サンプリング温度 | 1.0 |
| `--top-p` | top-p (nucleus) サンプリング閾値 (1.0で無効) | 1.0 |

### テスト

```bash
uv run pytest test_lm.py -v
```

## ファイル構成

| ファイル | 説明 |
|---|---|
| `recursive_compressor.py` | RecursiveCompressorモジュール（`step`/`forward`/`predict`） |
| `recursive_compressor_lm.py` | 言語モデル（PreTrainedModel継承） |
| `recursive_compressor_lm_pipeline.py` | パイプライン並列用ステージラッパー |
| `configuration_recursive_compressor.py` | モデル設定（PretrainedConfig継承） |
| `dataset.py` | HFデータセット読み込み・トークナイズ・memmapキャッシュ |
| `train_pipeline.py` | パイプライン並列学習スクリプト（Muon + AdamW、bfloat16 autocast） |
| `predict.py` | テキスト生成 |
| `predict_stream.py` | 対話的ストリーム生成 |
| `test_lm.py` | テスト |
| `.env.example` | 環境設定例 |

## 学習データセット（日本語のみ）

| データセット | 種類 |
|---|---|
| `wikimedia/wikipedia` (20231101.ja) | 文章 |
| `hotchpotch/cc100-ja-documents` | 文章 |
| `shi3z/ja_conv_wikipedia_llama2pro8b_30k` | 対話 |
| `shi3z/ja_conv_wikipedia_orion14B_100K` | 対話 |

文章データは `[DOC]` マーカー付き、対話データは `[QUERY]`/`[ANSWER]` マーカー付きでフォーマットされます。短い文書はパッキングして1サンプルにまとめ、PADによる無駄を削減しています。

## モデルパラメータ

| パラメータ | 値 |
|---|---|
| d_model | 2048 |
| num_heads | 16 |
| d_ff | 6144 |
| chunk_size | 4 |
| compress_size | 1 |
| num_layers | 16 |
| context_length | 2048 |
| optimizer | Muon (2D hidden) + AdamW (embedding/head/bias/norm) |
| learning rate | 5e-5 |
| precision | fp32 master weights + bfloat16 autocast |
