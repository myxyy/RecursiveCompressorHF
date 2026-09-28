# 標準LogKV + ゲート付きMamba-2枝：固定10桁実験

2026-09-28、直列ハイブリッドの結果をcommit `da7c8de`に保存し、
新ブランチ`mamba2-logkv-gated`でユーザー依頼の構成を実装した。
**9月28日08:11:19 JSTに両タスク各50,000 steps・全164セルの評価・自動監査が完了した。**
04:48:40 JSTにGPU0/1で開始し、本実行は3時間22分39秒。GPUは解放済み。
完了後、CPUで予測の再集計、重み・凍結ソース・現行ソース・対照データとの照合を再実行した。
[完了再確認・追加診断](experiments/mamba2-logkv-gated-20260928/completion-recheck.json)。
Copyingは直列版より改善したが、標準LogKVの長距離完全一致は維持できなかった。
Selectiveは訓練範囲付近で高精度になったものの、Mamba-2単体を超える長距離外挿は得られなかった。
[開始記録](experiments/mamba2-logkv-gated-20260928/launch.json)。
開始確認ではCopyingが200 steps、Selectiveが300 stepsまで正常に進行し、gateの更新、
事前測定との設定一致、凍結ソースのハッシュを確認した。
[開始確認](experiments/mamba2-logkv-gated-20260928/start-check.json)。

## 完了結果

以下は各256例の完全一致数で、各欄は **best / final**。
今回のbestはCopying 16,300 step、Selective 45,600 step、finalはともに50,000 step。
bestは訓練ログで選んでおり、外挿評価の結果から選び直してはいない。
対照の各bestもそれぞれの訓練ログから選んだ重みで、同一stepではない。

### Copying

| T | 今回：ゲート付き | 標準LogKV | Mamba-2単体 | 直列版 |
|---:|---:|---:|---:|---:|
| 2,048 | 256 / 256 | 256 / 256 | 256 / 256 | 256 / 256 |
| 4,096 | 256 / 256 | 256 / 256 | 255 / 255 | 248 / 248 |
| 6,144 | 252 / 233 | 256 / 256 | 154 / 175 | 96 / 96 |
| 8,192 | 85 / 16 | 256 / 256 | 27 / 39 | 2 / 2 |
| 12,288 | 0 / 0 | 256 / 256 | 2 / 0 | 0 / 0 |
| 131,072 | 0 / 0 | 256 / 256 | 0 / 0 | 0 / 0 |

今回の両checkpointはT≤4,096の全評価点で256/256。bestはT6,144で252/256まで保持したが、
T12,288以降の全評価点ではbest/finalとも完全一致0だった。
標準LogKVは同じ41点すべて256/256であり、目的とする長距離保持の維持には至っていない。

T131,072の桁精度はbest **28.59%**、final **11.52%**。
bestでは第1桁が247/256（96.48%）で、残り9桁は合計21.05%。
finalは256例すべてに同一の10桁列を出力した。bestでは217種類の出力があり、
一部情報は残っているが、10桁全体の保持とは大きく異なる。

![Copying比較](experiments/mamba2-logkv-gated-20260928/copying/comparison.png)

### Selective Copying

| T | 今回：ゲート付き | 標準LogKV | Mamba-2単体 | 直列版 |
|---:|---:|---:|---:|---:|
| 1,024 | 256 / 256 | 10 / 19 | 256 / 256 | 256 / 227 |
| 2,048 | 255 / 254 | 14 / 30 | 256 / 256 | 256 / 205 |
| 3,072 | 248 / 192 | 10 / 29 | 256 / 246 | 240 / 138 |
| 4,096 | 198 / 68 | 4 / 11 | 240 / 216 | 191 / 88 |
| 6,144 | 24 / 1 | 11 / 19 | 107 / 61 | 67 / 11 |
| 8,192 | 0 / 0 | 3 / 11 | 26 / 19 | 23 / 2 |
| 12,288 | 0 / 0 | 0 / 11 | 1 / 2 | 1 / 0 |
| 131,072 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

bestはT≤1,536の全評価点で256/256。標準LogKVに対する短距離・訓練範囲付近の改善は大きい。
一方、T6,144ではbestでも24/256、T8,192以降の全評価点でbest/finalとも0となり、
Mamba-2単体に対する外挿の優位性は得られていない。直列版bestとの比較もT3,072～4,096では
わずかに改善する一方、T6,144以降は悪化し、一貫した改善ではない。

T131,072の桁精度はbest **22.46%**、final **20.20%**。
bestの第1桁は232/256（90.63%）だが、残り9桁は合計14.89%。
Mamba-2単体bestの桁精度15.08%を上回っていても、10桁を一様に保持できるようになったとは言えない。
標準LogKVは同Tで41.05% / 48.24%であり、桁精度でも今回の構成を上回る。

![Selective比較](experiments/mamba2-logkv-gated-20260928/selective-copying/comparison.png)

### ゲートと学習の安定性

以下はFP32 master重みに対する`tanh(g)`の訓練ログ値。BF16評価時の丸め後の値は
各タスクの`review.json`に別途保存し、checkpointと照合済み。

| タスク | checkpoint | 平均絶対値 | 最小 | 最大 |
|---|---|---:|---:|---:|
| Copying | best | 0.00952 | −0.03527 | 0.03175 |
| Copying | final | 0.01524 | −0.08414 | 0.07004 |
| Selective | best | 0.02625 | −0.14770 | 0.16009 |
| Selective | final | 0.02820 | −0.13873 | 0.16690 |

両タスクともゲートは0から開いた。Selectiveの平均絶対値はCopyingより大きいが、
枝の特徴量の大きさや方向を測っていないため、係数をそのまま寄与率として解釈できない。
小さい係数も、標準埋め込みに対して小さい摂動になる保証はない。

Copyingは24,000 stepの訓練区間では完全一致100%だったが、24,100で15.41%、
24,200で13.64%に急落した。同時にゲート平均絶対値も0.00939から0.01320へ変化している。
その後訓練精度は回復したものの、finalの長距離性能はbestより低い。
Selectiveもbest直後の45,800 stepで区間完全一致82.97%に落ち、ゲート平均絶対値が
0.02627から0.02815へ増加した。finalの訓練区間完全一致は99.94%まで回復したが、
T4,096の評価はbest198/256に対してfinal68/256。

これらは**訓練範囲での回復と外挿性能の維持が一致しない**ことを示す。
ただし途中の外挿評価や介入はしていないため、ゲート変化が劣化の原因だとは断定できない。
AdamWのlrはwarmup後3e-4固定で、後半の学習不安定性も今後切り分ける対象となる。

![訓練推移と長距離の桁別精度](experiments/mamba2-logkv-gated-20260928/training-and-digits.png)

[Copyingゲート推移](experiments/mamba2-logkv-gated-20260928/copying/gate-history.png)、
[Selectiveゲート推移](experiments/mamba2-logkv-gated-20260928/selective-copying/gate-history.png)。

### 解釈と次に切り分ける点

元のLogKV入力経路を残すことは、今回のCopyingの中距離性能には有効だった。
しかし、全重みを共同学習する限り、初期gate0だけでは長距離記憶の解を維持できなかった。
「訓練範囲で有用なMamba特徴に合わせてLogKVの圧縮・読出しも適応し、外挿でその組合せが崩れる」
という仮説とは整合するが、状態やattentionの介入実験で確認した機構ではない。
今回の512係数は入力に依存しないため、それ自体に不要トークンを書き込まない選択機構はなく、
Mamba枝を加算するだけでは選択と長期保存の役割分担は保証されない。

次に試すなら、まず既存checkpointでゲート倍率0 / 0.5 / 1の短い評価を行い、
Mamba枝の寄与を確認するのが小規模な診断となる。ただし推論時にgateを0にしても、
共同学習済みのLogKV重みは元の標準LogKVには戻らない。別途、標準LogKVの学習済み重みを使い、
それを凍結してMamba枝だけを学習する対照で、重みの変化と入力の変化を切り分けられる。
これも入力を変えるため性能維持の保証ではない。長距離保持を必須にするなら、
入力加算以外に、既存の記憶経路を保持したまま選択的な書込みを学習させる設計を検討したい。

今回は各構成1 seedでパラメータ数も異なるため、構造そのものの限界や有意差は主張しない。
追加GPU評価・再学習・16M/1B延長は実行していない。

全数値は[Copying](experiments/mamba2-logkv-gated-20260928/copying/metrics.json) /
[Selective](experiments/mamba2-logkv-gated-20260928/selective-copying/metrics.json)、
予測・正解・マージンは同ディレクトリの`best.json.gz` / `final.json.gz`。
[全164セル監査](experiments/mamba2-logkv-gated-20260928/review.json)と
[CPU追加診断](experiments/mamba2-logkv-gated-20260928/diagnose_completed.py)も保存した。

## 構成

元のLogKV埋め込みと独立したMamba-2枝を、最初のLogKVBlockの前で合流する。

```
tokens ── LogKV embedding ────────────────────┐
                                             + → LogKVBlock ×2 → RMSNorm → LogKV head
tokens ── Mamba embedding → Mamba-2 ×2 → norm ─×
                                             ↑
                                          tanh(g)
```

`x_t = E_LogKV(token_t) + tanh(g) ⊙ MambaFeatures_t`。
`g`は512チャネルごとの学習パラメータで、全要素0で初期化する。
実効係数は−1..1の符号付きスケールで、入力ごとのsigmoid gateではない。
最初の更新ではgateに勾配が流れ、Mamba枝の重みの勾配は0になる。
gateが開くとMamba枝も学習する。branchを計算から除外してgateの学習を妨げる実装にはしていない。

- 標準LogKVの埋め込み、各層のCausalConv4、attention、FFN、最終norm、**非共有の出力head**を維持。
- Mamba-2枝は独立した埋め込み・公式初期化・2層・幅4の因果畳み込み・最終RMSNormを使用する。
  枝内部のLM headは埋め込みと共有され、追加パラメータは持たないが、最終出力計算には使用しない。
- d_model512、LogKV2層・8 heads・d_ff1024・C4、Mamba2層・d_state128・expand2・headdim64・ngroups1。
  LogKVの位相埋め込みなし、固定レベル減衰、gate/self slotあり。総パラメータ数**9,238,624**。
- LogKVを先に構築してからMamba枝を構築するため、gate0では同じseedで構築した現行標準LogKVと
  全共有パラメータ・出力が一致する。これは過去の対照実験の初期重みとの一致を主張するものではない。
- 学習済み重みは転用せず、両枝とgateを共同学習する。gate0での一致は、学習後の性能維持の保証ではない。
- hiddenはMambaの固定サイズ状態と、CausalConv履歴を含むLogKVの階層状態。合計は系列長の対数サイズ。
  分割入力の境界で状態や勾配を切り離さない。

[直列版](mamba2-logkv-hybrid.md)ではLogKVの入力全体をMamba出力に置き換え、LogKV側convを外していた。
今回は元の入力経路・conv・headを維持してMambaを加算する。ただし学習中にgateが開けば入力表現は変わり、
LogKVの重みも更新されるため、元の長距離性能が自動的に保護されるわけではない。

## 実験条件

Copying / Selective Copyingを別々にseed0から各50,000 steps学習する。
固定10桁、T=1..2,028 loguniform、batch64 / accum1、AdamW lr3e-4 / warmup1000 / weight_decay0、
勾配clip1、全位置の位置合わせCE、FP32 master重み＋BF16 autocast。
訓練100-step区間の完全一致率・桁精度・EMA lossでbestを選び、finalと合わせて評価する。

評価は各タスク41点（T=1..14、2の累乗と中間点、最大131,072）×256例×best/final、計164セル。
seed12345、BF16重み＋autocast、CPUデータ生成、8,192トークンずつ状態を引き継ぐ。
従来と同じ評価データで標準LogKV・Mamba-2単体・直列版と比較する。
各モデルのパラメータ数・構成・初期化が異なるため、同パラメータ数の対照比較ではない。

ゲート512要素の`tanh(g)`、最小・最大・平均・平均絶対値を100 stepsごとに保存する。
best/final重みと履歴・評価時のBF16 gateをCPUで照合し、ゲートの推移も図示する。
ゲートの大きさだけからMamba枝の因果的な寄与を断定しない。

GPU0でCopying、GPU1でSelectiveを並列実行した。最大2GPUで連続実行の許可範囲内。
各タスク300-step測定と全41点×8例の評価器smokeを通過した。
実測はCopying 0.239秒/step、Selective 0.246秒/step。学習30%の余裕と評価30分を含め、
事前見積もりは4.85 / 4.98時間、終了目安09:45 JST頃、停止期限11:44:36 JSTだった。
実際は08:11:19 JSTに終了し、最終事前測定開始04:44:36からも約3時間27分で完了した。
本2タスク・best/final評価・CPU解析で終了し、16M/1B延長・追加seedは実行していない。

## 検証

既存Mamba-2・直列版・新構成の19テストと、追加のGPU gate0一致テストが通過（計20件）。
gate0で標準LogKVとCPU出力・初期パラメータがbit一致し、GPU BF16 autocast出力もbit一致した。
gateを開いた状態でCPU FP64の分割出力・勾配・状態の一致、状態の非破壊、保存量を検証。
GPU FP32 / autocast / BF16重みで公式Mamba学習経路と分割推論の一致、FP32勾配の一致も確認した。
HF保存・再読込、モデルディレクトリ指定の推論、キャッシュ付きgenerateが通過した。

本番サイズbatch64・T2028で2回の更新を確認。全パラメータの勾配が有限で、gateが0から更新された。
ピーク割当は22,016,784,384 bytes（約20.50 GiB）。
本番評価のバッチ上限付近（T2048で253例、T8192で63例）も通過し、出力は有限だった。
評価ピーク割当は12,697,687,552 bytes（約11.83 GiB）。
全41点の評価器smokeで状態量・予測からの正答数を確認し、gateの訓練ログ・重み・BF16評価値も一致した。
両タスクの全41点×256例の正解列、Selectiveの配置位置がLogKV対照と一致することをCPUで確認した。

直列版の初回測定で経験した断片化を避けるため、最初から`PYTORCH_ALLOC_CONF=expandable_segments:True`を使用する。

## 使用方法と保存先

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
uv run --extra mamba2 python -m exp.copying.train --arch mamba2-logkv-gated \
  --run-name gated-copying --d-model 512 --num-layers 2 --mamba-num-layers 2 \
  --num-heads 8 --d-ff 1024 --conv-kernel-size 4 \
  --steps 50000 --max-t 2028 --t-dist loguniform --batch-size 64 \
  --lr 3e-4 --warmup 1000 --seed 0 --eval-interval 0

# Selectiveはexp.selective_copying.trainに変更する。
uv run --extra mamba2 python -m exp.copying.evaluate --run-name gated-copying \
  --checkpoint best --max-t-exp 17 --samples 256
```

モデルは`models/mamba2_logkv_gated/`、キャンペーンと予測保存付き評価器は`exp/mamba2_logkv_gated/`。
共通のモデル読込は`mamba2_logkv_gated`を識別する。今回の重みは数字10語彙の実験用。
大型成果物・凍結ソースは`/mnt/raid0/RecursiveCompressor/experiments/mamba2-logkv-gated-20260928-run/`。
各タスク配下の`campaign.json`が進行状態、`train.log`が学習ログ、`results/{best,final}.json`が評価結果。
[事前確認・解析コード・成果物](experiments/mamba2-logkv-gated-20260928/)を参照。
