# LogKV アーキテクチャ 知見まとめ

`logkv` ブランチで開発中の新アーキテクチャ **LogKV** について、設計・実装・
実験から得られた知見の記録。実装は [logkv.py](../logkv.py)（`LogKV` /
`LogKVBlock`）、[logkv_lm.py](../logkv_lm.py)（`LogKVLM`）、
[configuration_logkv.py](../configuration_logkv.py)、訓練は
[train_logkv.py](../train_logkv.py)、生成は [predict_logkv.py](../predict_logkv.py)、
テストは [test_logkv.py](../test_logkv.py) / [test_logkv_lm.py](../test_logkv_lm.py)。

## 1. 背景と動機

RecursiveCompressor 系のモデルは自然言語を学習できたが、生成が進むと特定の話題語に
執着する現象（`text/oil.txt` の「オイル」反復）が解消できなかった。copying-memory
ブランチで位置情報（chunk_pos_emb → CausalConv）を整備しても執着は残り
（[copying-memory-branch-changes.md](copying-memory-branch-changes.md)）、
「固定の学習ベクトル（initial_context / compressor_query / pos_emb）を残差ストリームに
繰り返し注入する構造」自体との相性を疑って、main のアーキテクチャには手を入れず
新設計に切り替えた。

LogKV の設計方針:

- **kv はすべて内容由来**。学習済み固定ベクトルの注入なし、位置埋め込みなし
- 各位置が参照する kv の数を **O(C·log L)** に抑えつつ、受容野は全系列
- 逐次推論の hidden state も **O(C·log L·d)**（系列長に対して対数）

## 2. アーキテクチャ定義

### 2.1 圧縮階層

q, k, v = 線形射影（bias なし）。**マルチヘッド**: ヘッドをバッチ次元に折り畳んで `(B·H, L, d_head)` とし、階層圧縮と窓 attention のコア（`_attend`）をヘッドごとに独立に 通した後、ヘッド結合 → 出力射影 `lo`（bias なし）。hidden のバッチ次元も B·H になる。スケールは d_head^(−1/2)。`Compressor` はチャンク
（長さ C = chunk_size）を **チャンク末尾の q をクエリとした attention プーリング** で
1 要素に圧縮する（q_out = 末尾 q、k_out / v_out = softmax(q_out·kᵀ) による凸結合）。

レベル i の列（サブユニット = C^i トークンの要約）を C 個ずつチャンク化して圧縮すると
レベル i+1 の列になる。列長が C 以下になった残り列を **最上位レベル** として保持する
（当初の設計では捨てられており受容野が約 C^levels の直近窓に限られていたため、
0e40bbd で追加。これにより L ≤ C でも前トークン attention が成立する）。

### 2.2 attention のスロット意味論

位置 s のクエリは、レベル i（サブユニット長 C^i、ブロック長 C^(i+1)）ごとに C 個の
スロットを見る。s がブロック u のサブユニット j にいるとき:

- スロット c < j → 自ブロック u の完成済みサブユニット c
- スロット c ≥ j → 前ブロック u−1 のサブユニット c（自ブロック側は未完成 or 局所未来）

全レベルのスロットを連結して **単一の softmax** で正規化し value を加重和する。
有効スロットが 0 の系列先頭は出力 0（`nan_to_num`）。

### 2.3 レベル減衰ロジットバイアス（3f7cc40、§5.3 参照）

レベル i のロジットに `i · log(1/C)` を加算する（スロット重みに C^(−i) を掛けるのと
等価）。パラメータフリー。

## 3. 実装上の知見

### 3.1 step() が唯一の実装、forward/predict は委譲

`step(x, hidden)` は任意長セグメントを **トークンループなし** で一括処理する
（ループはレベル数 O(log L) のみ）。成立する根拠は次の 3 性質:

1. **ブロック整列のスロット選択は連続窓と等価**: 「c<j は自ブロック / c≥j は前ブロック」
   を展開するとクエリのサブユニット直前の連続 C 個のサブユニットになる。スロット順は
   閉形式 `a_c = q_sub − C + ((c − q_sub) mod C)` で再現でき、gather で完全ベクトル化
   できる（スロット順まで一致するため forward と数値まで同一）
2. **窓内要素は必ずクエリより前に完成する**（要素 a ≤ q_sub−1 の完成トークンは
   クエリより前）ので、階層状態をレベル一括で先に更新してから全クエリの attention を
   まとめて計算しても causal を破らない
3. **必要な過去は prev + cur が正確にカバーする**: セグメント最古クエリの窓開始は
   直前完成チャンク（prev）の先頭以降になる

hidden は `(levels, offset)`。`levels[i] = [cur_q, cur_k, cur_v, prev_k, prev_v]`
（構築中チャンク <C 要素 と直前の完成チャンク）、`offset` は処理済みトークン数
（絶対位置の復元に必要）。呼び出し側の hidden は破壊しない。

forward は `step(x)` の一発呼び出し、`predict(x (B, d), hidden)` は 1 トークン版
（RecursiveCompressorAttention と同インターフェース）。

### 3.2 等価性の担保

forward が step 委譲になると step==forward のテストが自明化するため、当初の並列実装
（レベル構築 → ブロック整列 gather）を [test_logkv.py](../test_logkv.py) の
`reference_forward` に独立オラクルとして保持している。step / 分割 step / 1 トークン
predict 連鎖は fp64 で参照実装と **機械精度（<1e-12）** で一致する。意味論を変更する
とき（レベル減衰など）は参照実装にも同じ変更を入れる。

### 3.3 LM 化

`LogKVBlock` = pre-norm の標準ブロック（`x += LogKV(RMSNorm(x)); x += FFNSwiGLU(RMSNorm(x))`）。
`LogKVLM` は RecursiveCompressorLM と同じ骨格（embedding std 0.02、RMSNorm、head、
all-PAD ラベルの NaN ガード、HF `generate()` 用に per-layer hidden を `past_key_values`
として opaque に持ち回り、Cache ラップ無効化）。greedy `generate()` が step+argmax の
逐次生成と完全一致することをテストしている。

### 3.4 訓練インフラ

LogKVLM は 1 GPU に収まるので **DDP データ並列**（`torchrun --nproc_per_node=6
train_logkv.py`）。既存の mmap キャッシュ（`hf_cache/mmap/ctx2048/`, pretrain）を
そのまま利用。Muon（2D 隠れ層）+ AdamW（embedding/head/1D）、線形 warmup、bf16 autocast、
勾配蓄積、control.cmd、`--resume latest`、TensorBoard（`tensorboard/logkv-pretrain/`）。
rank 0 が `--sample-interval` ごとに固定日本語プロンプト 3 種から生成し `samples.log`
に追記する（文法獲得の目視確認用）。

実測（d1024 / d_ff 3072 / 16 層 / C=4 / ctx2048、シングルヘッド 294M パラメータ。
8 ヘッド版は `lo` の分 +17M）:

- batch 2/GPU で 16 GiB、**batch 4 は OOM**。支配項はレベルごとの slot gather
  `(L, C·levels, d)` の保存
- 約 4,000 tok/s/GPU、6 GPU DDP 実効 **約 15k tok/s**（1000 ステップ ≈ 50 分）
- 8 ヘッド版（310M）: batch 2/GPU で 15.9 GiB、約 4,260 tok/s/GPU（シングルヘッドと同等）

## 4. 実験 1: 日本語文法の獲得（補正なし、run `d1024-l16`）

設定: 上記モデル、batch 2 × accum 2 × 6 GPU = 実効 24（約 49k トークン/ステップ）、
lr 2e-4、warmup 1000、seed 0。

| step | EMA loss | 生成の様子（temperature 0.7, top-p 0.9） |
|---|---|---|
| 100 | 10.6 | — |
| 1000 | — | 助詞・活用・文末形・括弧年号の書式が出現（内容は無意味）。「お」反復ループあり |
| 5000 | 3.62 | 連体修飾・並列句・文をまたぐ話題維持まで成立。反復ループ消滅 |
| 6277 | 3.50 | save_and_exit で停止（checkpoint-6277 をベースラインとして保持） |

注: 実験 1・2 は **シングルヘッド**（マルチヘッド化前、出力射影なし）の結果。
マルチヘッド化（`lo` 追加、`num_heads` 設定）以降のコードではこれらのチェックポイントは
読み込めない（`lo` 欠落）。再現には 3f7cc40 時点のコードを使う。

**結論: LogKV は日本語文法を問題なく獲得できる**（位置埋め込み・固定ベクトルなしで）。

## 5. 実験 2: 話題執着の同定と対策

### 5.1 症状（checkpoint-6277、1024 トークン生成）

3 プロンプトすべてで「序盤は正常 → 序盤に出た内容語が支配的になる → 末尾でその語の
反復に崩壊」。生成テキストを 4 等分した出現回数:

| プロンプト | 語 | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|---|
| 日本の首都は | 車 | 24 | 29 | 12 | 1 |
| | 形 | 0 | 17 | 27 | **142** |
| 昔々あるところに | 酒 | 14 | 24 | 40 | **54** |
| 人工知能とは | 生物学 | 21 | 25 | 9 | 1 |
| | 分子 | 2 | 0 | 30 | **68** |

序盤語の頻度が単調増加し、末尾は「形形形形…」「分子の分子の分子…」。支配語が
車→形、生物学→分子と **乗り換わりながら** それぞれ増殖する。EOS で終端した生成は
0/3（執着の正帰還で文書を終えられない）。

### 5.2 原因仮説: レベル横断の多重計上

Compressor は softmax プーリングなので、**顕著なトークンの value は希釈されずに上位
レベルへ生き残る**。距離 d のトークンはレベル i₀ ≈ log_C d から最上位までの
約 (log L − log d) 個のスロットに **全濃度で同時に存在** し、softmax 内で実出現回数の
log 倍に多重計上される。これが「出現 → スロット占有率増 → attention が引かれる →
再出現」の正帰還を増幅する。最上位スロットは一度完成すると以後の全クエリから見え続ける
ため、序盤内容の引力が永続する。予測「序盤語が進行とともに増える」は 5.1 で確認された。

### 5.3 対策: レベル減衰ロジットバイアス

レベル i のロジットに `i·log(1/C)` を加算（重み C^(−i)）。同一トークン由来のレベル横断
コピー群の重みが `C^(−i₀)(1 + 1/C + 1/C² + …)` の等比和に潰れ、**実質 1 回分の計上** に
正規化される。実際に何度も出現した語が高頻度に見える「正しいカウント」は保存される。

副次的性質:

- 距離 d の最安アクセスレベルが log_C d なので **~1/d のべき乗則 recency prior** を
  誘導する（パラメータフリー・スケール不変、固定位置埋め込みを避ける方針と整合）
- 最上位スロットの重みが C^(−top) ≈ 1/L に減衰し、序盤内容を思い出すには内容マッチで
  log L 程度のロジット差を跳ね返す必要がある形になる

### 5.4 効果（run `d1024-l16-leveldecay`、同条件で 5000 ステップ、checkpoint-5000）

| | 補正なし（6277 步） | レベル減衰（5000 步） |
|---|---|---|
| 1024 トークン完走時 | 3/3 が単一語反復に崩壊 | 崩壊はデータ様式的な反復 2 件のみ |
| 序盤語の四分位推移 | 全サンプルで単調増殖 | 日本 1→1→0→1、社会 2→1→0→1 と平坦 |
| EOS 自然終端 | 0/3 | **7/9**（seed 0–2 × 3 プロンプト） |
| 5000 步 EMA loss | 3.62 | 3.69 |

seed 1・2 は全プロンプトで一貫した短い文章を書いて `</s>` で閉じる。**文書を「書き
終えられる」ようになったこと自体が、序盤内容の永続的引力が消えた証拠**。

残った症状（seed 0）: 「衆議院議員」のリスト反復（議員 3→4→9→18。ただし Wikipedia
の一覧・フッター様式の模倣で、自力で EOS 終端）と全角スペース連続（cc100-ja に実在する
パターン）。いずれも旧機構ではなく小規模 LM に一般的な同一トークン連発の範疇で、
repetition penalty が効くタイプ。

loss が 0.07 高い点は recency prior による長距離参照のハンディか揺らぎか、このステップ数
では判別できない。

## 6. 未解決・今後の候補

- **長距離検索の回帰確認**: レベル減衰は遠い内容の参照に log d のロジット逆風を課す。
  `exp/copying` / `exp/selective-copying` を LogKVLM 対応にして長 T の string accuracy
  を測る（旧アーキでは T≈25k まで外挿できていた）
- **長期訓練での執着再発確認**: 20000 ステップ程度まで回し、1024 トークン生成の四分位
  分析を再実施。loss 差（3.69 vs 3.62）が縮むかも確認
- 減衰係数の可変化: `−i·β`（β を per-layer 学習、init log C）は必要になれば容易
- メモリ削減: slot gather `(L, C·levels, d)` の保存が VRAM の支配項。レベルごとに
  逐次 softmax する形（online softmax）にすれば batch を増やせる
- 残る反復（リスト様式・空白連続）への repetition penalty 適用と chat_server 対応

## 7. コマンド

```bash
uv run pytest test_logkv.py test_logkv_lm.py -v
uv run torchrun --nproc_per_node=6 train_logkv.py --run-name <name> --max-steps 5000
uv run torchrun --nproc_per_node=6 train_logkv.py --run-name <name> --resume latest
uv run python predict_logkv.py --model-dir $DATA_DIR/checkpoints_logkv/<name>/checkpoint-5000/model \
    --max-new-tokens 1024 --seed 0
```

チェックポイント: `$DATA_DIR/checkpoints_logkv/d1024-l16/checkpoint-6277`（補正なし）、
`$DATA_DIR/checkpoints_logkv/d1024-l16-leveldecay/checkpoint-5000`（レベル減衰）。
