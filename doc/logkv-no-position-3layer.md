# 位置埋め込みなしLogKV：2層から3層への比較

2026-09-13、進行中の[2層の再評価](logkv-no-position-main.md)に対し、ユーザーの指定で
3層のCopying / Selective Copyingを追加する。**本学習は開始準備中、結果は未確定。**
追加実験の見積もりは評価込み約6.3時間。事前GPU検証を含む停止期限は
2026-09-14 07:03:24 JST。

## 比較条件

ここでの層数は積層する`LogKVBlock`の数（`num_layers`）を指す。
各Block内部のKV圧縮は従来どおり再帰的で、圧縮階層の数を3に制限する変更ではない。
凍結ソースは2層実験と共通のmain `0932d8c`で、全27ファイルが`3b0ce51`と一致する。
mainのモデル・通常訓練コードは変更しない。

| 項目 | 2層対照 | 今回の3層 |
|---|---|---|
| num_layers | 2 | 3 |
| パラメータ数 | 5,786,112 | 8,673,792（+2,887,680） |
| GPU | 0=Copying、1=Selective | 2=Copying、3=Selective |
| C / d / heads / d_ff | 4 / 512 / 8 / 1024 | 同左 |
| 位相埋め込み / 追加位置表現 | 無効 / なし | 同左 |
| gate / self slot / 固定−i log C | 有効 | 同左 |
| KV/V norm・level amplification | 無効 | 同左 |
| 記憶長・学習 | 固定10桁、各50k steps、batch64、grad accumulation1 | 同左 |
| T分布・損失 | loguniform[1,2028]、全位置CE | 同左 |
| 最適化 | AdamW、lr3e−4、warmup1000、weight decay0、clip1 | 同左 |
| seed | 初期化0、訓練データ1、評価12345 | 同左 |
| 精度 | fp32重み＋bf16 autocast学習、bf16重み＋autocast評価 | 同左 |
| 評価 | best/final、T131072まで41点、各256例 | 同左 |

`phase_levels=2`は無効な位相埋め込みの設定値として残るだけで、埋め込みは生成しない。
bestは訓練100 steps区間のstring/token/−EMA lossの辞書順で選ぶ。
タスクごとに独立したモデルを初期状態から学習する。

3層モデルをseed0で生成後、embedding・最初の2層・最終norm/headの初期重みを
2層対照の保存済み**未学習**重みと全テンソル一致させる。追加した第3層の重みだけを
3層モデルの新規初期化から残す。この3層初期重みを両タスクで共用する。
学習済み重みの転移は行わない。訓練データは専用generatorを使い、各タスクの
2層・3層でseed・batch・乱数消費手順を揃える。異なるタスク同士のT系列は同一ではない。

## 検証と評価方針

最大訓練長T2028（系列長2048）・batch64で2回のoptimizer更新を確認した。
ピークallocatedは約18.3 GiB、reservedは約18.9 GiBで、batchを変更せずに実行できる。
各タスク20 stepsをGPU2/3で反復し、重みと記録指標のbit一致を確認した。
決定論モードと`CUBLAS_WORKSPACE_CONFIG=:4096:8`を使う。
短い反復は50k全体・異なる環境でのbit再現性を保証しない。

300 stepsの同時実測はCopying 98.6秒、Selective 86.6秒。
50kへの単純外挿は4.56時間 / 4.01時間で、学習時間15%余裕・評価1時間・事前検証を
加えた見積もりは6.29時間。GPU0/1の2層学習も稼働中の実測値。
実checkpointの両タスクbest/finalをT3/16/65で動作確認し、独立generatorによる
記憶列・配置再生と予測の正答数照合を通過した。

- [事前学習・時間計測](experiments/logkv-no-position-3layer-20260913/preflight.json)
- [最大訓練長のメモリ確認](experiments/logkv-no-position-3layer-20260913/fullsize_smoke.json)
- [評価器確認](experiments/logkv-no-position-3layer-20260913/evaluation_smoke.json)

完了後、全164セルの正答数・桁別誤答数・実際の記憶配置をCPUで再検証する。
2層対照も監査済みなら、同じ例の予測を照合し、完全一致数・各桁の誤答数に加えて
「2層で誤答→3層で正答」と逆方向の例数を保存する。
2層対照が未完了なら比較を保留として記録し、追加GPU実験は開始しない。

過去の[位置縮退の診断](logkv.md#62-t1-異常の原因-同一トークン連続区間での位置縮退確認済み)は
重複あり構造の結果である。現在の構造で同じ桁が厳密に識別不能だとは扱わない。
今回の比較は現在の構造における経験的な改善を調べるもので、3層で縮退が必ず解消するという
仮定は置かない。学習seedは1個で、層数とともにパラメータ数も約50%増えるため、
改善しても容量増加と独立した純粋な深さの効果とは断定できない。

## 実行管理

今回に限り追加2GPUを使用し、2層対照と合わせて最大4GPU。
2層のGPU0/1と元の停止期限は維持し、3層はGPU2/3の別supervisorで管理する。
追加実験の事前GPU検証は9月13日23:33:24 JST開始。7.5時間で停止し、
失敗時はこの追加実験のworker・子プロセスを停止する。追加seed・位置方式・16M延長はない。

大型生成物：`/mnt/raid0/RecursiveCompressor/experiments/logkv-no-position-3layer-20260913/`。
[実行コード一覧](experiments/logkv-no-position-3layer-20260913/README.md)。
