[English](README_en.md) | 日本語

# RecursiveCompressor / LogKV

各方式の比較と再現用commitは[実験記録一覧](doc/logkv-experiments.md)にまとめています。
標準CausalConvでの[可変桁実験](doc/logkv-variable-memory.md)も完了しました。可変桁で学習したCopyingは
10桁・T=131,072で238/256例が完全一致しましたが、32・64桁は全評価セルで完全一致に届きませんでした。

2026-09-14、各LogKVBlockのattention前に幅4のCausalConvを加え、位置埋め込みを使わない構成を標準に採用しました。
新規訓練CLIはConv幅4・gate/self slot有効が既定です。`--conv-kernel-size 0`でConvを無効化できます。[実装・比較実験](doc/logkv-causal-conv.md)を参照してください。
位置埋め込みなし・固定10桁の評価では、CopyingはT=131,072までの全41点で各256/256例が完全一致しました。
追加のT=16,777,216でも8例中8例が完全一致しました（固定10桁、学習seedは1個）。
Selective Copyingも改善しましたが、評価した最長距離T=131,072での完全一致は未達です。

階層的なkv圧縮による独自アーキテクチャ **LogKV** の言語モデル実装です。

![LogKVのKVキャッシュ構造（重複なし）](logkv-refine.drawio.png)

## アーキテクチャ（LogKV）

LogKVは、系列をチャンク（C=chunk_size）単位で再帰的にattentionプーリング圧縮し、各クエリ位置が各階層の**現在チャンク内の完成済み部分だけを、単一のsoftmaxで参照**するattention機構です。過去全体を重複も欠落もない区間に分割し、各階層で最大C−1個のkvスロットを使います。例えばC=4の位置4では要約0〜3だけ、位置5では要約0〜3とトークン4を参照します。

- 受容野は全系列、attentionあたりのkv数は O(C·log L)
- 逐次推論の隠れ状態も O(C·log L·d)（系列長に対して対数）。各階層では未完成チャンクだけを保持します
- `forward` / `step`（隠れ状態持ち回りの任意長チャンク処理）/ `predict`（1トークン）が fp64 で機械精度一致するよう実装・テストされています

標準構成は以下の要素からなります（検証記録は [doc/logkv.md](doc/logkv.md)）:

| 要素 | 内容 |
|---|---|
| レベル減衰 | レベルiのロジットに −i·log C。従来の粗い階層へのペナルティを維持。重複除去後の有効性は再検証予定 |
| CausalConv | 各Blockのattention前に幅4のdepthwise causal convolution残差を追加。位置埋め込みは使用しない |
| マルチヘッド | ヘッドをバッチ次元に折り畳んで適用 |
| Gated attention | sigmoid(W_g x) を各ヘッドのattention出力に乗算 |
| Self slot | クエリ自身のトークンのk/vを1スロット追加（通常のcausal maskと同じ意味論）。softmaxに「聞かない」逃げ場を与え勾配を安定化 |

言語モデル（LogKVLM）は `Embedding → LogKVBlock × num_layers → RMSNorm → Linear` で、HuggingFaceの `PreTrainedModel` を継承しています（`save_pretrained` / `from_pretrained` / `generate` 対応）。

**従来の2階層の位置埋め込みを用いた重複なし構造の実験（Copyingの記憶長M=10固定）では、訓練ホライズン2,028に対し、CopyingのT=131,072までの全41評価点で完全一致率100%**を確認しました（各256例、best/finalとも）。そのfinalチェックポイントによる追加のT=16,777,216でも8例中8例が完全一致しました。これは過去の構成での結果であり、実験ブランチの代替位置表現で達成した結果ではありません。一方、同条件の旧構造との比較ではSelective Copyingの精度が低下しています（[実験詳細](doc/logkv-refine-experiments.md)、学習seedは各条件1個）。新構造のLM生成品質は未評価です。旧チェックポイントの重みは読込可能ですが、新構造での出力は変わります。

2026-09-13、位置表現の調整は一旦区切りとし、mainの既存アーキテクチャでの訓練へ戻る方針としました。
実験ブランチからはドキュメント・結果・実験コードのみを取り込み、モデル・設定・訓練CLIは変更していません。
[Copying](doc/logkv-aligned-rope.md)・[Selective Copying](doc/logkv-aligned-selective.md)を含む
[全実験の一覧と再現方法](doc/logkv-experiments.md)を参照してください。
位置表現の実験専用フラグはmainでは利用できません。

2026-09-14、位置埋め込みを無効にした[2層の再評価](doc/logkv-no-position-main.md)と
[3層との比較](doc/logkv-no-position-3layer.md)が完了しました。3層化で短～中距離Copyingと
Selective Copyingの桁精度は改善しましたが、Copyingの長距離完全一致は回復せず、
長距離の桁精度は低下しました（固定10桁、各50k steps、各構成1 seed）。

## セットアップ

```bash
uv sync
cp .env.example .env
# .env の DATA_DIR を編集（データセット・チェックポイントの保存先）
```

## 使い方

### 学習（DDPデータ並列）

```bash
uv run torchrun --nproc_per_node=6 train_logkv.py \
    --run-name myrun --conv-kernel-size 4 --gated-attention --self-slot
```

混合精度（fp32マスター重み + bfloat16 autocast）、Muon（隠れ層の2D重み）+ AdamW の2段オプティマイザで学習します。attention部はonline softmax + activation checkpointingでVRAMを削減しています。

学習データはHuggingFaceから自動ダウンロードされ、トークナイズ済みキャッシュ（numpy memmap）が `$DATA_DIR/hf_cache/mmap/ctx{context_length}/` に保存されます。チェックポイントは `$DATA_DIR/checkpoints_logkv/{run-name}/` に保存され、`--resume latest` で再開できます（消費済みデータをスキップして継続、`--max-steps` は絶対ステップ数）。1000ステップごとに日本語プロンプトからのサンプル生成が `samples.log` に記録されます。

#### 学習中の制御

```bash
just pause          # 一時停止（プロセス維持・GPU idle）
just resume         # 再開
just save-and-exit  # チェックポイント保存して終了 → --resume latest で再開
```

### テキスト生成

```bash
# 1回生成
uv run python predict_logkv.py --model-dir $DATA_DIR/checkpoints_logkv/myrun/checkpoint-5000/model \
    --max-new-tokens 1024 --temperature 0.7 --top-p 0.9 "日本の首都は"

# 対話的にストリーム生成（config.json からアーキテクチャを自動判別）
uv run python predict_stream.py --model-dir /path/to/checkpoint \
    --context-length 4096 --temperature 0.7 --top-p 0.9
```

### テスト・基礎実験

```bash
uv run pytest test_logkv.py test_logkv_lm.py -v   # LogKV（fp64機械精度の等価性検証を含む）
uv run pytest test_lm.py -v                       # 旧アーキテクチャ

# Copy Memory Problem / Selective Copying（長距離記憶の基礎検証）
uv run python exp/copying/train.py --arch logkv --conv-kernel-size 4 --gated-attention --self-slot \
    --run-name myrun --t-dist loguniform
uv run python exp/copying/evaluate.py --run-name myrun --max-t-exp 17
```

## ファイル構成

| ファイル | 説明 |
|---|---|
| `logkv.py` | LogKVアーキテクチャ本体（`forward`/`step`/`predict`、LogKVBlock） |
| `logkv_lm.py` | 言語モデル LogKVLM（PreTrainedModel継承） |
| `configuration_logkv.py` | モデル設定（PretrainedConfig継承） |
| `train_logkv.py` | DDPデータ並列学習スクリプト（Muon + AdamW、bfloat16 autocast） |
| `predict_logkv.py` | テキスト生成（LogKV用） |
| `predict.py` / `predict_stream.py` | テキスト生成・対話的ストリーム生成（新旧アーキ自動判別） |
| `dataset.py` | HFデータセット読み込み・トークナイズ・memmapキャッシュ |
| `test_logkv.py` / `test_logkv_lm.py` | LogKVのテスト |
| `exp/copying/`, `exp/selective-copying/` | 長距離記憶の基礎実験一式 |
| `doc/logkv.md` | LogKVの設計・実験の知見まとめ |
| `.env.example` | 環境設定例 |

### 旧アーキテクチャ（RecursiveCompressor）

再帰的な圧縮・展開でチャンク間の情報伝達を行う旧実装も残っています: `recursive_compressor.py` / `recursive_compressor_lm.py` / `recursive_compressor_lm_pipeline.py` / `configuration_recursive_compressor.py` / `train_pipeline.py`（6GPUパイプライン並列、`uv run torchrun --nproc_per_node=6 train_pipeline.py`）。経緯は [doc/copying-memory-branch-changes.md](doc/copying-memory-branch-changes.md) を参照してください。

## 学習データセット

`--dataset-type` で選択:

### `pretrain` (文書データ)
| データセット | 言語 |
|---|---|
| `wikimedia/wikipedia` (20231101.ja) | 日本語 |
| `wikimedia/wikipedia` (20231101.en) | 英語 |
| `hotchpotch/cc100-ja-documents` | 日本語 |
| `JeanKaddour/minipile` | 英語 |

### `instruct` (対話データ)
| データセット | 言語 |
|---|---|
| `shi3z/ja_conv_wikipedia_llama2pro8b_30k` | 日本語 |
| `shi3z/ja_conv_wikipedia_orion14B_100K` | 日本語 |
| `HuggingFaceH4/ultrachat_200k` | 英語 |

データ形式はLlama 2スタイルで、文書は `<s>text</s>`、対話は `<s>[INST]q[/INST]a</s>` （ターンごとにBOS/EOSで囲む）です。長文は context_length 単位で分割し、短文は連結してパッキングすることで PAD による無駄を削減しています（対話データは会話間パッキングなし・応答のみloss）。

## モデルパラメータ（LogKV標準構成）

| パラメータ | 値 |
|---|---|
| d_model | 1024 |
| num_heads | 8 |
| d_ff | 3072 |
| chunk_size | 4 |
| num_layers | 16 |
| context_length | 2048 |
| phase_emb | 無効 |
| conv_kernel_size | 4 |
| self_slot | 有効 |
| gated_attention | 有効 |
| optimizer | Muon (2D hidden) + AdamW (embedding/head/bias/norm/Conv) |
| learning rate | 2e-4（線形warmup 1000） |
| precision | fp32 master weights + bfloat16 autocast |

旧checkpointの設定読込と低水準APIの既定値は互換性のため維持します。`LogKVConfig`で直接標準モデルを作る場合は`conv_kernel_size=4, gated_attention=True, self_slot=True, phase_emb=False`を指定してください。
