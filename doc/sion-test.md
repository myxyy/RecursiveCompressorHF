# sion_test.py（Needle-in-a-Haystack）の LogKV での実行記録

友人提供の評価パイプライン [sion_test.py](../sion_test.py) を LogKV（[doc/logkv.md](logkv.md)）
で実行した記録。ブランチ `logkv-sion-test`。

## 1. スクリプトの概要

- **タスク**: 合成 Needle-in-a-Haystack。`[10, vocab−100)` の一様乱数トークン列（長さ L）の
  `[0.1L, 0.8L)` の位置に `key=7777, value∈[1000, 5000)` を隣接して埋め込み、末尾を
  `query=9999` にして、末尾位置の logits から value を当てる（4000 クラス分類）
- **訓練**: カリキュラム L=32 → 512、各 300 ステップ、batch 16、AdamW lr 3e-4、
  訓練精度 >98%（step>100）で早期終了
- **評価**: L ∈ {512, 1024, 2048, 4096, 8192}、各 20 試行
- **ラッパー** `LanguageModel`: `nn.Embedding(16000, 256)`（デフォルト init N(0,1)）を
  `lm_head` と tied、LayerNorm → head。バックボーンは `(B, L, 256) → (B, L, 256)` の任意の
  モジュール

## 2. 変更点

`sion_test.py` はプレースホルダ `backbone = #MODEL INITIALIZATION`（構文エラー）を
LogKV バックボーンで埋めた以外は無変更:

```python
LogKVBackbone(HIDDEN_DIM)   # LogKVBlock × 2 層, 4 ヘッド, d_ff 512, chunk_size 4,
                            # phase_emb (levels=2), gated_attention, 固定 log C レベル減衰
```

= LogKV の標準構成（doc/logkv.md §6.8）。1.45M パラメータ。

追加スクリプト:

- [sion_test_diag.py](../sion_test_diag.py): 切り分け用ドライバ。`sion_test.py` の関数を
  そのまま使い、init の変種とステップ数を引数で切り替え、訓練後に重みを
  `$DATA_DIR/exp/sion_test_{variant}_{steps}.pt` に保存
- [sion_test_eval.py](../sion_test_eval.py): 保存重みの長文評価。指標は
  `evaluate_needle_in_a_haystack` と同一だが、(a) 試行をトークン予算で分割し、
  (b) バックボーンを `step()` のチャンク逐次処理（hidden 持ち回り、一発 forward と数値的に
  等価）で走らせ、(c) norm/head は末尾位置だけに適用する（予測は同一）。ラッパーの
  全位置 logits `(B, L, 16000)` fp32 は L=2048 × 200 試行で 26 GB になり OOM するため

## 3. 結果

### 3.1 友人のプロトコルそのまま（各フェーズ 300 ステップ）: 全長 0%

```
Phase 32 : Loss 233 → 21.7, Train Acc 0%
Phase 512: Loss      → 20.0, Train Acc 0%
Context 512 / 1024 / 2048 / 4096 / 8192: 0.00% (0/20) すべて
```

訓練が成立していない。原因はラッパーの初期化: tied embedding が N(0,1) のため
**初期 logit std ≈ 16、初期 CE ≈ 248**（一様分布なら log 16000 ≈ 9.7）で、lr 3e-4 × 300
ステップでは logit スケールの正規化すら終わらない。

### 3.2 切り分け（`sion_test_diag.py`）

| 変種 | ステップ/フェーズ | 結果 |
|---|---|---|
| init そのまま | 300 | 訓練精度 0%、評価 0%（上記） |
| embedding init std 0.02 のみ変更 | 300 | 初期 CE 9.7 に正常化するが loss 8.4 で未収束、評価 0% |
| init そのまま | 2000 | **512 フェーズが step 466 で早期収束（>98%）** |

→ LogKV はタスクを学習できる。300 ステップという予算がこのラッパーに対して短すぎる
（init を直しても 300 では足りない。~1500–2500 ステップが必要）。

### 3.3 学習成立後の長文評価（init そのまま・2000 ステップ、200 試行）

| 文脈長 | 512（訓練長） | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 | 65536 | 131072 |
|---|---|---|---|---|---|---|---|---|---|
| 正答率 | 75.5% | 72.0% | 72.5% | 62.0% | 56.5% | 43.0% | 25.0% | 22.0% | 6.5% |

- 偶然は 1/4000 = 0.025%。訓練長の 16 倍（8192）で 56.5%、**256 倍（131072）でも 6.5% =
  偶然の 260 倍**。崖のない滑らかな減衰
- 域内が 75% に留まるのは、早期終了が「1 バッチ（16 サンプル）で >98%」を条件にしており
  収束が浅いため。ステップを増やせば全体が底上げされる見込み
- hidden state は文脈長に対して対数サイズ（レベル数 × C 要素 × d）なので、評価時のメモリは
  文脈長によらずチャンク幅で決まる（131072 でも問題なし）

## 4. 再現手順

```bash
git checkout logkv-sion-test
uv sync

# 友人のプロトコルそのまま（300 ステップ）
uv run python sion_test.py

# 切り分け: init そのまま 2000 ステップ + 簡易評価(20 試行)。重みを $DATA_DIR/exp/ に保存
uv run python sion_test_diag.py asis 2000 20
# embedding init 0.02 の変種
uv run python sion_test_diag.py emb002 300 20

# 保存重みの長文評価（200 試行、131072 まで）
uv run python sion_test_eval.py $DATA_DIR/exp/sion_test_asis_2000.pt 200 \
    512,1024,2048,4096,8192,16384,32768,65536,131072
```

環境: RTX 3090（24 GB）1 枚、PyTorch（`uv sync` の環境）、`tqdm` が必要
（無ければ `uv run --with tqdm python ...`）。訓練は 2000 ステップ × 2 フェーズで約 3.5 分、
131072 までの評価は数分。

## 5. 補足（タスクの性質）

このタスクは haystack が一様乱数トークンで、Copying の blank のような「捨てやすい内容」が
ない。LogKV の圧縮は attention プーリングで顕著な内容を残す仕組みなので、needle の
(key, value) が乱数トークンの海の中で圧縮を生き残る必要があり、doc/logkv.md の
Copying（1M トークンまで完全解）より厳しい設定。Selective Copying と同様の
「崖のない減衰」プロファイルになっている。
