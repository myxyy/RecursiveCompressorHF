# copying-memory ブランチ変更まとめ

Copy Memory Problem（人工データセット）による基礎検証と、そこから導かれた
アーキテクチャ改善の記録。検証の詳細な経緯・実験ログは
[doc/instruction-for-claude/copying-task.md](instruction-for-claude/copying-task.md) を参照。

## 背景と結論

**背景**: 自然言語の学習は安定してきたが、「遠距離の情報を正確に運べるか」という
基礎能力を Copy Memory Problem（CKConv, arXiv:2102.02611 の定式）で検証した。

**結論**: 検証開始時のアーキテクチャは **T=1（系列長21）ですらコピーを解けなかった**
（全介入が token acc ≈0.53 に収束する頑固な壁）。体系的な消去法と bag-order 分析で
「圧縮の値空間に順序情報が存在しない」ことを特定し、`chunk_pos_emb` の導入で解決。
最終的に d_model=512・2層・11M パラメータのモデルが:

- 訓練域内 (T≤2028): string accuracy 0.81〜1.00
- **訓練ホライズンの約12倍 (T≈25k) まで string accuracy ≈0.6**
- 48倍 (T≈98k) でも token accuracy 0.82

という長さ外挿を達成した（絶対位置埋め込みなしの構造で）。

## アーキテクチャ変更一覧（モデル本体）

### 1. attentionゲートの開放初期化 (177acf6)
- `MultiHeadAttention.gate_linear.bias` を +4.0（sigmoid≈0.98）で初期化
- 近ゼロ初期化のsigmoidゲートは全attention出力を約0.5倍し、再帰k階層で 0.5^k の
  信号減衰が複合するため
- 付随修正: HF `post_init()` の汎用 `_init_weights` が全Linearのbiasをゼロ化して
  サブモジュール `__init__` 内の手動初期化を上書きするため、
  `RecursiveCompressorLM._init_weights` のオーバーライドで適用

### 2. ALiBi相当のヘッド毎位置スロープ (f8a5f5b) → **後に削除**
- 学習可能な `compressor_query_pos` を削除し、固定・非学習のALiBiバイアス
  （スロープ 2^(-8h/H)）を attention logit に加算する方式へ
- **その後のablationで削除が決定**: chunk_pos_emb（変更6）導入後は位置情報として
  冗長であり、負スロープによる過去valueの希釈が支配的な弊害になることが判明
  （d512比較: 訓練域12倍のT≈25kまでno-ALiBiがstring accで+0.2〜+0.4の大差、
  ALiBiが勝るのは16倍以遠のgraceful degradationのみ）。位置情報は
  chunk_pos_emb（値空間タグ）に一本化し、encoderのcausalマスクは
  `mask_tril`（boolバッファ）に復帰
- 得られた知見: バッファはpersistentにすること（`persistent=False` だと
  `from_pretrained` のmeta-device初期化経路で未初期化になる）

### 3. decompressor kv の retrieve_size 窓 (7cbc6c4)
- 新ハイパーパラメータ `retrieve_size`（k、デフォルト4）を config に追加
- チャンク i の decompressor kv を「直前チャンクの圧縮1点」→
  「チャンク i-k..i-1 の圧縮列（k×compress_size 点）」に拡張
- `initial_context` を `(retrieve_size, compress_size, d_model)` に拡張（学習可能）。
  系列先頭では `cat([prev_outer, compressed_out])` からの unfold スライスにより
  initial_context との混合窓が特別分岐なしで得られる
- decompressor にチャンク距離のALiBiバイアス `-m_h*(k-1-w)` を追加
- hidden の outer は直近k点 `(B, k, S, d)`。メモリは O(k·log L) のままで無限生成可

### 4. compressorバイアスのクエリ別化 (d64e6b4)
- compress_size (S) > 1 のクエリはS本とも「チャンク末尾ベクトル由来の同一値+同一
  バイアス」で完全に縮退しており、S>1 が容量に寄与していなかった
- クエリ s に担当位置 `c(s) = s(C-1)/(S-1)` を割り当て `-m_h*|c(s)-j|` でバイアス。
  S=1 は従来の末尾距離バイアスとビット一致（後方互換）
- ※ S>1 は hidden が S 倍になる空間計算量とのトレードオフ（現運用は S=1）

### 5. 標準デコーダ順への処理順入れ替え (5de7631)
- 旧: 圧縮 → 再帰 → **decompressor(cross-attn) → encoder(self-attn)**
- 新: **encoder(self-attn) → 圧縮 → 再帰 → decompressor(cross-attn)**
  （標準Transformerデコーダブロックの self-attn → cross-attn 順）
- cross-attnのクエリが位置分化済みの表現になり、同一トークンの連続領域でも
  位置別の検索が可能に
- comp_query の導出も encoder 出力のチャンク末尾（=チャンク全体のcausal要約）に変更

### 6. chunk_pos_emb — 値空間の位置タグ（決定打）(5de7631)
- `chunk_pos_emb (chunk_size, d_model)`（学習可能、std=0.02）をチャンク化直後の
  トークンに加算
- **ALiBiはattentionの重みしか歪めず、圧縮 z = Σ softmax·W_v e_j の値空間に
  位置が刻まれない**。このため圧縮ベクトルは中身の bag（多重集合）を完全に保持する
  一方で順序を完全に失う（実測: bag復元率100%、順序はbag-only理論最適値0.412と
  厳密一致 — 壁0.53 = (2×1.0+8×0.412)/10 の正体）
- 値空間への位置タグにより順序が圧縮を生き残るようになり、T=1 が1500ステップで
  100%に到達。**チャンク内C個のみ・全レベル共有・長さ非依存**

## 処理フローの before / after

```
before:  chunk化 → [comp_query導出(生埋め込み)] → 圧縮 → 再帰
         → decompressor(kv=直前圧縮1点, クエリ=生埋め込み) → encoder → 出力

after:   chunk化 → +chunk_pos_emb → encoder(causalマスク)
         → [comp_query導出(encoder出力チャンク末尾)] → 圧縮 → 再帰
         → decompressor(kv=過去k点の窓, クエリ=encoder出力) → 出力
```
(ALiBiバイアスは一時導入されたがablationの結果削除。位置情報はchunk_pos_embのみ)

## 実験基盤（exp/copying/）(08eb71e, 8ba5755, 2b7bbe2)

- `task.py`: CKConv定式のCopy Memory Problem生成・採点（フォーマットテスト12件付き）
- `train.py`: 単GPUオンライン生成訓練。`--t-dist loguniform`（重要:
  T~U[1,2028]の一様サンプリングでは answer 信号が全位置lossの~1%と薄く短Tの足場も
  出現しないため学習が立ち上がらない）、ベスト重み保存（`model_best/`、学習の
  一時的不安定化で最終checkpointが谷に当たる事故への保険）
- `evaluate.py`: チャンク分割step推論（フルforwardと等価、hiddenはO(log L)）で
  T=1..131072 の汎化カーブを評価。`--checkpoint {auto,best,final}`
- 結果の置き場: `$DATA_DIR/exp/copying/{run_name}/`（results.json, plot.png 等）

## 診断プロセスの要約（消去法の記録）

| 仮説 | 介入 | 結果 |
|---|---|---|
| 記憶容量不足 | compress_size 1→8 | 0.516 不変 |
| ゲートの信号減衰 | 開放初期化 | 0.530 微改善のみ |
| 位置情報の欠如(重み側) | ALiBi導入 | 0.528 不変 |
| 層の分業不足 | 2→3, 4層 | 0.530 不変 |
| 検索粒度の粗さ | retrieve_size窓 | 0.5295 不変 |
| クエリ縮退 | クエリ別バイアス | 0.5295 不変 |
| クエリの位置盲目 | デコーダ順入替 | 0.5295 不変 |
| **順序が値空間に無い** | **chunk_pos_emb** | **1500stepで100%** |

決め手は per-digit 分析: 1チャンク1桁の出力は100%、1チャンク4桁は0.41
（= bag-only最適値0.4121と一致）→「bagは完璧、順序がゼロ」という機構的特定。

## 互換性・運用上の注意

- **旧checkpointと非互換**: `compressor_query_pos`/`mask_tril` の削除、
  `initial_context` の形状変更、`chunk_pos_emb`/ALiBiバッファの追加。
  本ブランチのモデルは新規訓練が必要
- config に `retrieve_size` が追加された（旧configの読み込みは `getattr` の
  デフォルト4でフォールバック）
- hidden state の outer が `(B, S, d)` → `(B, k, S, d)` に変更
  （chat_server / predict 系は hidden をopaqueに扱うため変更不要）
- step/forward/predict の数値等価性は全変更を通じて維持（test_lm.py 62件 +
  exp/copying 12件パス）

## 未解決・次の課題

- T≥32k で string accuracy が段階的に低下（〜6k / 〜25k / 〜100k の段差は
  再帰レベル境界と対応する可能性 — 未解析）
- Mamba Fig.2 の次のタスク **Selective Copying**（content-aware選択が必要）
- 本番言語モデル訓練への反映（本ブランチのアーキ変更が言語モデリング性能に
  与える影響の確認）
