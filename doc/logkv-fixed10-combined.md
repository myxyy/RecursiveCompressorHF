# combined-no-decay：10桁固定のCopying / Selective Copying

## 状態と実行範囲

2026-09-08、GPU 0・1を各タスク1台ずつ使用して並列実行し、完了した。
再実行の開始から全評価終了まで10:48:30〜14:40:59 JST、約3時間52分。
各50,000ステップ、best/finalそれぞれ41個のT・各256例、合計164評価セルを保存した。
GPUは解放済み。追加評価・再学習は続行確認まで開始せず、
16,777,216トークン評価と本ブランチのmainへのマージは保留している。

CopyingはT≤3,072の**評価した30点**で完全一致率100%になったが、
T=4,096で1/256例、T≥8,192の評価点では0/256例だった。
Selective CopyingはfinalのT=64で46.09%に改善した一方、T≥1,024の評価点では
best/finalとも完全一致0%。固定10桁でも長距離の保持は未解決である。

## 目的

[可変記憶長の比較](logkv-position-study.md)で比較的良かったcombined-no-decayを、
[従来の固定桁実験](logkv-refine-experiments.md)と同じM=10・追加prefixなしで再学習し、
記憶を維持できる待ち時間Tを調べる。可変Mのチェックポイントの追加評価ではない。

## 条件

| 項目 | 設定 |
|---|---|
| アーキテクチャ | 重複なしLogKV、d=512、8 heads、d_ff=1024、2 layers、C=4 |
| 位置表現 | compressor position transform + relative K/V |
| 減衰・旧位置埋め込み | level_decay_scale=0、phase_emb=false、scalar RPEとcompressor logit decayも無効 |
| その他 | gated attention、self slot、KV/V normなし |
| 記憶長・prefix | M=10固定、P=0 |
| 学習 | 各50,000 steps、batch 64、gradient accumulation 1、T~loguniform[1,2028] |
| 最適化 | AdamW、lr=3e-4、warmup=1000、weight decay=0、gradient clip=1 |
| 損失 | 全位置のposition-aligned CE、解答を入力へ戻さない |
| seed | 学習0、学習データ1、評価12345 |
| best選択 | 100 stepsの訓練区間string accuracy、token accuracy、−EMA lossの辞書順 |
| 評価 | best/final、41個のT（1〜131,072）、各256例 |
| 精度 | 学習fp32 weights + bf16 autocast、評価はモデル重みもbf16（旧固定桁実験と同じ） |
| パラメータ数 | 5,802,496（旧phase2は5,794,304） |

学習・評価プログラムは旧固定桁実験と同じ`exp/{task}/train.py` / `evaluate.py`。
実装commitは`24b360cf85712fde3ee7a2da4d61e2eb45350a51`の凍結worktreeを使用する。
旧phase2とのrun_config差分は、方式に関する新旧フラグ、パラメータ数、run名だけ。
seed番号は一致するが、異なるモデル構成間で共通初期重みが同一であることは保証していない。
各条件1 seedであり、学習のばらつきは評価していない。

可変M実験との間にはM/P以外にも、gradient accumulation、データ生成乱数の消費順、
best選択方法、評価時の重みの精度などの差がある。そのため両実験の精度差を
M固定化だけの因果効果と解釈しない。

## 実行記録

- [実行スクリプト](experiments/logkv-fixed10-combined-20260908/run.py)
- 保存先：`/mnt/raid0/RecursiveCompressor/experiments/logkv-fixed10-combined-20260908`
- 最大T=2028・batch=64のforward/backward事前確認：ピークallocated 19.65 GiB。
- 初回はSelective CopyingでCUDA allocatorの予約領域の断片化が疑われるOOMが発生。
  両タスクを停止し、初回のログ・モデル・manifestを`attempt1-allocator-oom/`へ保存した。
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`を設定し、同じseedで最初から再実行。
  バッチサイズと学習条件は変更していない。
- 実行段階の上限は8時間。到達時には停止して続行確認を必要とする。


## 学習とチェックポイント

| Task | 学習時間（分） | best step | final EMA loss |
|---|---:|---:|---:|
| Copying | 211.69 | 50,000 | 3.3140e-7 |
| Selective Copying | 215.63 | 49,000 | 0.011566 |

Copyingのbestはfinalと同じ50,000ステップで、safetensorsのSHA256と全41評価点の
結果が一致する。独立な2回の成功を意味しない。Selectiveはbestとfinalが異なる。
ここでのbestは**訓練区間の指標による選択**であり、各Tでの評価精度が最良とは限らない。

![学習曲線](experiments/logkv-fixed10-combined-20260908/learning.png)

## Copying：固定10桁の再生はできるが、深い階層への外挿はできていない

以下はbest/final共通、各256例。旧phase2は全41点でtoken/stringとも100%だった。

| T | token % | string % | 完全一致例数 |
|---:|---:|---:|---:|
| 1〜3,072の30評価点 | 100.00 | 100.00 | 各256/256 |
| 4,096 | 62.77 | 0.39 | 1/256 |
| 6,144 | 63.09 | 0.78 | 2/256 |
| 8,192 | 63.79 | 0.00 | 0/256 |
| 16,384 | 38.67 | 0.00 | 0/256 |
| 131,072 | 33.44 | 0.00 | 0/256 |

T≥8,192の全9評価点で完全一致例はなかった。可変M実験の同方式ではM10・T64の
完全一致率は75.39%だったのに対し、今回の固定条件では100%になった。
10桁の順序再生そのものは学習できたが、旧phase2で確認した長距離保持を再現していない。

「3,072以下のすべてのTで成功する」とは解釈できない。50,000ステップの簡易評価では、
T=507 / 1,014 / 2,028 / 4,056 / 8,112の完全一致例数はそれぞれ
64 / 54 / 64 / 64 / 0（各64例）だった。T=1,014のように短い側にも失敗がある。
簡易評価は学習時のfp32重み＋bf16 autocast、標準評価はbf16重みであり、
サンプル数と乱数系列も異なるため、双方の差をTだけの効果とは断定しない。

C=4で4,096=4^6、訓練中の最大系列長は2,048なので、4,096トークンをまとめた
上位チャンクは訓練中に登場しない。T=4,056の簡易評価では成功し、T=4,096の
標準評価で急落したことは、追加の圧縮階層への外挿が課題である可能性を示す。
ただし、破綻する厳密な境界や、原因がCompressor・retrieval・精度のどこにあるかは
この評価だけでは特定していない。

## Selective Copying：短いTの一部で改善、長距離は未達

全てstring accuracy（%）、各256例。旧方式の数値は固定M10のrefined phase2実験。

| T | 今回best | 今回final | 旧phase2 best | 旧phase2 final |
|---:|---:|---:|---:|---:|
| 1 | 100.00 | 100.00 | 100.00 | 99.22 |
| 16 | 93.75 | 95.31 | 72.27 | 56.64 |
| 32 | 77.34 | 88.67 | 39.06 | 30.86 |
| 64 | 3.12 | 46.09 | 17.58 | 12.89 |
| 96 | 29.30 | 31.64 | 10.16 | 12.11 |
| 128 | 1.95 | 28.12 | 8.20 | 3.13 |
| 512 | 0.39 | 1.56 | 1.95 | 1.95 |
| 2,048 | 0.00 | 0.00 | 0.78 | 0.39 |
| 131,072 | 0.00 | 0.00 | 0.00 | 0.00 |

finalのT64は118/256例、token accuracyは91.76%。bestの同点は8/256例なので、
チェックポイント依存が大きい。bestのT1・2・4、finalのT1・3は256/256例が完全一致したが、
T≥1,024の全15評価点ではbest/finalとも完全一致例はない。
T131,072のtoken accuracyもbest 12.81%、final 12.27%で、
旧phase2より短距離では改善する点がある一方、長距離の保持は改善していない。

![旧phase2との比較](experiments/logkv-fixed10-combined-20260908/comparison.png)

点線は訓練T上限2,028。線は評価点をつないだもので、点間のすべてのTを評価したわけではない。
Copyingのbest/finalは同一重み・同一評価結果なので重なっている。

## 検証・成果物

- 2タスク×学習・best評価・final評価の6コマンドがすべて正常終了。
- 各タスク500件の訓練区間ログ、10件の簡易評価、best選択、設定フラグを検証。
- 各checkpointの41個のT、n=256、token/stringの正答数との整合性を検証。
- Copyingのbest/final重みのSHA256および全結果の一致を検証。
- GPU追加使用を伴う実装変更・追加実験は行っていない。

[成果物一覧と再現手順](experiments/logkv-fixed10-combined-20260908/README.md)、
[全比較CSV](experiments/logkv-fixed10-combined-20260908/comparison.csv)、
[集計JSON](experiments/logkv-fixed10-combined-20260908/summary.json)、
[代表点のtoken/string表](experiments/logkv-fixed10-combined-20260908/tables.md)を参照。
初回OOMのログも別フォルダに保存し、成功した再実行の結果と区別している。

## 追加の境界評価（2026-09-08、ユーザーの続行確認後に実施）

CopyingのfinalでT=4,064〜4,112の49点を1刻み・各256例で評価した。
GPU0のみ、再学習なし。14:49:10〜14:52:15 JST、評価本体184.63秒。
標準評価と同じbf16重み＋bf16 autocast、`step()`、8192トークン単位、token budget=2^19。
重みのSHA256が前回finalと一致することを確認した。

今回はTごとにgeneratorをseed=12345へ戻し、**全49点で同じ256個の10桁列**を使用した。
標準41点評価ではgeneratorを連続使用していたため、同じTでも評価例は一致しない。
全Tのターゲット列の一致と、各例の予測からtoken/string/各桁の正答数を再計算して検証した。

| T | 完全一致例数 | string % | 観測 |
|---:|---:|---:|---|
| 4,064〜4,076（各点） | 256/256 | 100.00 | 全10桁が正解 |
| 4,077 | 188/256 | 73.44 | 最後の1桁だけに誤り |
| 4,078 | 11/256 | 4.30 | 最後の2桁に誤り |
| 4,079 | 29/256 | 11.33 | 最後の3桁に誤り |
| 4,080 | 0/256 | 0.00 | 最後の4桁に誤り |
| 4,086 | 0/256 | 0.00 | 先頭桁も境界到達。先頭予測は全256例でblank 0 |
| 4,096 | 1/256 | 0.39 | 全桁が境界以降 |
| 4,112 | 1/256 | 0.39 | 全桁が境界以降 |

解答の第j桁（j=1〜10）の0始まり位置は`p = T + 9 + j`。
T=4,077では第10桁がp=4,096、T=4,078では第9桁がp=4,096となる。
今回の範囲では**p<4,096の出力44,800個がすべて正解**で、誤りはp≥4,096に限られた。
同じ256例を各Tで繰り返した集計であり、44,800個の独立な記憶課題を意味しない。
また、今回の範囲外のTや他seedでの保証ではない。

![境界評価と各桁の正答率](experiments/logkv-fixed10-combined-20260908/boundary/boundary.png)

白い破線は各桁の解答位置がp=4,096になる点。失敗の開始がこの線に沿っている。
C=4の重複なし構造では、p=4,096で過去の[0,4,096)が上位1チャンクとして参照される。
訓練中の最大系列長2,048では登場しない追加の圧縮階層であり、
**失敗開始とその階層への切替が一致する**ことを確認した。
圧縮での情報損失、上位の表現を読み出す際の不一致、数値精度の寄与はまだ分離していない。

### バッチサイズの交絡確認

標準の自動バッチサイズもT=4,076→4,077で128→127に切り替わるため、
同じ2点だけバッチサイズを入れ替えて確認した（GPU0、追加計算7.95秒）。
T=4,076をbatch=127、T=4,077をbatch=128としても、
全256例・全10桁の**予測そのものがそれぞれ元の評価と一致**した。
したがって、この2点の急落をバッチサイズ変更で説明することはできない。
また全系列長は8192未満で、`step()`呼び出しの分割境界も通過していない。

### 成果物と停止位置

[評価スクリプト](experiments/logkv-fixed10-combined-20260908/boundary.py)、
[バッチ確認スクリプト](experiments/logkv-fixed10-combined-20260908/check_boundary_batch.py)、
[集計・図の再生成](experiments/logkv-fixed10-combined-20260908/summarize_boundary.py)、
[全49点の結果](experiments/logkv-fixed10-combined-20260908/boundary/results.json)、
[各例のターゲットと予測](experiments/logkv-fixed10-combined-20260908/boundary/answers.json)、
[バッチ確認結果](experiments/logkv-fixed10-combined-20260908/boundary/batch_check.json)、
[各桁の正答数CSV](experiments/logkv-fixed10-combined-20260908/boundary/metrics.csv)を保存した。
GPUは解放済み。ここで追加評価を終了し、再学習・追加アブレーション・16M評価は開始していない。
mainへのマージも保留を維持する。

その後、ユーザー承認の[圧縮表現診断](logkv-compression-diagnostic.md)を完了した。
phase2との実K/V・Query・logit比較では、combinedの記憶差分の縮小と方向変化を観測し、
最初の未学習圧縮段の失敗はfp32でも再現した。U0単独の因果検証と16M再評価は未実施。
