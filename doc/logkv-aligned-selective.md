# 基準位置を揃えたRoPEのSelective Copying比較

2026-09-13、[Copying比較](logkv-aligned-rope.md)の終了・全300セルの監査後、
ユーザーの続行依頼に従い、未実施のSelective Copyingを別の実験として実行した。
**両方式の50k stepsと全132評価セルが、2026-09-13 21:36:04 JSTに正常終了した。**
本実行は17:19:05–21:36:04 JST、4.283時間。
GPU事前確認開始17:14:55からは4.352時間（約4時間21分）で、7.5時間の上限内に収まった。
準備commit `a597467`、開始記録commit `3674c9a`。GPUはすべて解放済み。
両方式の本学習の最初の100 stepsは、事前benchmarkの指標とbit一致した。
[開始と初期区間の確認記録](experiments/logkv-aligned-selective-20260913/launch.json)。
ブランチ`logkv-aligned-rope`で作業し、mainにはマージしない。

## 結果

**基準位置を揃えたRoPEに、一貫したSelective Copyingの改善は見られなかった。**
短距離では条件間の優劣がcheckpointによって変わり、長距離の桁正答率は新方式が低かった。
以下は各Tで256例の完全一致数。best / finalを併記する。

| T | local-control | aligned |
|---:|---:|---:|
| 1 | 256 / 255 | 255 / 256 |
| 3 | 216 / 232 | 230 / 212 |
| 16 | 107 / 99 | 79 / 108 |
| 64 | 17 / 28 | 25 / 17 |
| 128 | 21 / 14 | 10 / 11 |
| 256 | 3 / 1 | 2 / 1 |
| 512 | 6 / 3 | 1 / 2 |
| 1,024 | 1 / 0 | 0 / 0 |
| 2,048 | 1 / 2 | 0 / 0 |
| 4,096 | 0 / 0 | 0 / 0 |
| 8,192 | 0 / 0 | 0 / 0 |

![標準評価の完全一致率](experiments/logkv-aligned-selective-20260913/results/comparison.png)

完全一致が0でも、桁ごとの保持性能には差がある。次は回答2,560桁の正答率（%、best / final）。

| T | local-control | aligned |
|---:|---:|---:|
| 64 | 76.05 / 80.00 | 79.96 / 79.14 |
| 1,024 | 59.06 / 64.77 | 52.54 / 54.65 |
| 2,048 | 56.99 / 61.99 | 44.80 / 44.80 |
| 4,096 | 47.62 / 56.64 | 21.02 / 20.08 |
| 8,192 | 46.29 / 53.63 | 19.34 / 17.89 |

T8192のalignedでは各桁に広く誤りが出ている。
例えばbestの1桁目は45/256、10桁目は81/256正解で、途中の桁も29–57/256。
同じ評価例のlocal-control bestは1桁目241/256、10桁目149/256である。
新方式の問題を一箇所の桁ずれだけで説明することはできない。
T8192の予測列はbestが255種類、finalが256種類あった。
保存KVの情報損失と読み出しの失敗の切り分けは、この評価では行っていない。

best選択stepはlocal-controlが49,000、alignedが48,100。
選択された訓練100 steps区間の完全一致率は44.609% / 43.078%、最終区間は39.656% / 36.797%。
どちらもSelective自体の学習が完全一致に達しておらず、訓練範囲内のT1024でも新方式の桁正答率が低い。
したがって、域外の大きな回転角だけを原因と断定することもできない。

Copyingでは新方式のT131072がbest/finalとも0/256となり、今回のSelectiveでも長距離性能は改善しなかった。
この倍率1・固定モデルサイズの比較では、局所読み出しRoPEを置き換える根拠は得られていない。
回転合成の位置的整合性は、圧縮された記憶の保存・読み出しを学習できることの保証ではない。
ただし1学習seedでの結果であり、RoPE周波数の別設定、有限区間への写像、複数モーメントを使う方式を
否定する実験ではない。bestは訓練区間の指標で選択しており、評価値での最良checkpointではない。
両タスクは別モデルで、同一重みが両タスクを処理する評価でもない。

[全132セル・桁別誤答数](experiments/logkv-aligned-selective-20260913/results/metrics.csv)、
[保存予測・実配置・学習ログ](experiments/logkv-aligned-selective-20260913/results/README.md)、
[終了記録](experiments/logkv-aligned-selective-20260913/campaign.json)。

## 完了確認

独立CPU監査で、全132セル・337,920回答桁の正答数と桁別誤答数を照合した。
評価seed12345と元のminibatchサイズで凍結Selective生成器を再生し、33点すべての記憶列・
配置位置が全方式・checkpointの保存サンプルと一致することを確認した。
推論自体の再実行は行っていない。

36ソース・8実行スクリプト・3事前検証記録・40結果ファイルのハッシュ、
初期重みと全4チェックポイント、RAIDとアーカイブの一致、5,786,112パラメータの形状、
各500訓練ログ区間とbest選択も検査に通過した。
学習ログの所要時間はlocal-controlが14,369.1秒、alignedが15,313.9秒。
best/final評価の合計はそれぞれ82.72秒、86.86秒だった。

[独立監査記録](experiments/logkv-aligned-selective-20260913/results/manual_review.json)、
[再検証スクリプト](experiments/logkv-aligned-selective-20260913/review.py)。
元の自動監査`results/review.json`と実行済みスクリプトは変更していない。
予定したCopying・Selective両比較は完了し、追加のGPU実験は起動していない。

## 比較条件

Copyingと同じ2条件を、それぞれ初期状態からSelective Copyingで学習する。

| 項目 | local-control | aligned |
|---|---|---|
| 位置表現 | 局所読み出しRoPE、倍率1 | 末尾基準を揃えた圧縮・読み出しRoPE、倍率1 |
| phase2 / self slot / gate | なし / あり / あり | 同左 |
| 減衰 | 固定−i log C | 同左 |
| モデル | C4、d512、8 heads、d_ff1024、2層、5,786,112 parameters | 同左 |
| 学習 | 固定10桁、50k steps、batch64、T loguniform[1,2028]、全位置CE | 同左 |
| 最適化 | AdamW、lr3e−4、warmup1000、weight_decay0、clip1 | 同左 |
| seed | 初期化0、学習データ1 | 同左 |

凍結ソースはCopyingで使用したcommit `8dcfea4`を再利用し、全36ファイルのSHA256を確認する。
学習開始時には、Copyingの共通初期重みと全テンソルが一致することを検査する。
Copyingの学習済み重みからの転移学習ではない。
重み・訓練ログ・評価結果は`exp/selective-copying/`以下の独立したrunに保存する。

10個の数字を先頭T+9位置にランダムに配置し、11個のmarkerに続く回答位置で出現順を再現する。
共有trainer/evaluatorを呼び出す前に、凍結Selectiveタスクの実体を明示的に読み込み、
タスク名・ソースパス・SHA256も保存する。データ生成の追加乱数消費は行わない。
両方式のデータ乱数は揃えるが、配置抽選のあるSelectiveとCopyingでは乱数消費順が異なるため、
両タスクの各stepのT系列まで同じとはしない。

`torch.use_deterministic_algorithms(True)`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、
cuDNN benchmark=False。学習はfp32重み＋bf16 autocast、評価はbf16重み＋autocast。
過去の非決定的設定の結果とは区別し、今回再学習するlocal-controlを主対照とする。

## 評価と検証

- 各方式best/final、従来の標準33点、T8192まで各256例、計132セル。
- 評価seed12345をT間で継続し、従来のminibatchサイズと乱数消費順を維持する。
- bestは訓練100 steps区間のstring/token/−EMA lossで選び、評価値では選ばない。
- 各例の実際の配置位置・記憶列・10桁の予測・logit marginを保存する。
- 完了後、CPUで同じ凍結generatorを再生し、全セルの記憶と配置を照合する。
  正答数・桁別誤り・学習50k完遂・best選択・設定・重みハッシュも確認する。
- Copyingで用いた固定記憶34境界点は、このSelective比較には追加しない。

既存のモデル327テスト合格記録は凍結ソースのハッシュ一致を確認して継承する。
Selectiveタスクの既存11テストは今回再実行して合格。
GPUでは両方式それぞれ20 stepsをGPU0/1で繰り返し、重み・記録指標の一致を調べる。
300 stepsの同時実測から50kの時間を見積もり、その実checkpointで短いTの評価経路も確認する。

## 事前確認の結果

GPU0/1で各方式20 stepsを繰り返し、重み・記録指標のbit一致を確認した。
本学習と同じ設定で300 stepsを同時に測定した結果：

| 方式 | 300 steps | 50k学習への単純外挿 |
|---|---:|---:|
| local-control | 89.8秒 | 4.157時間 |
| aligned | 92.0秒 | 4.259時間 |

15%の余裕、評価30分、GPU事前確認を含めた見積もりは5.43時間。
短い区間からの予測であり、終了時刻を保証する値ではない。
[測定記録](experiments/logkv-aligned-selective-20260913/preflight.json)、
[モデル検証の継承とタスクテスト](experiments/logkv-aligned-selective-20260913/validation.json)。
300 stepsの実checkpointから、両方式best/finalをT3/16/65・各256例で評価し、
予測の集計と元generatorの記憶・配置の再生成照合も通過した。
[評価器の動作確認](experiments/logkv-aligned-selective-20260913/evaluation_smoke.json)。

## 実行制約と記録

最大2GPU、GPU0=local-control、GPU1=aligned。
学習見積もりに15%の余裕、標準評価30分、GPU事前確認時間を加えた全体時間を用いる。
8時間以上になる見込みなら本学習前に確認を取る。実行器はより短い7.5時間を上限とし、
GPU事前確認開始から計測する。失敗または期限超過時は両workerと子プロセスを停止する。
この実験の後に追加seed、別位置方式、16M評価を自動で実行しない。

大型生成物：`/mnt/raid0/RecursiveCompressor/experiments/logkv-aligned-selective-20260913/`。
凍結ソースと初期重み：`/mnt/raid0/RecursiveCompressor/experiments/logkv-aligned-rope-20260913/`。
[実行スクリプト](experiments/logkv-aligned-selective-20260913/README.md)。
