# mainの重複なし構造：位置埋め込みなしの再評価

2026-09-13、ユーザーの指定により、mainのアーキテクチャで位置埋め込みを無効にした
固定10桁Copying / Selective Copyingを再学習・評価する。モデル本体の変更はない。
**2026-09-13 22:18:53 JSTにGPU0/1で本学習を開始した。今回の結果は未確定。**
両タスクの最初の100 stepsは、事前benchmarkと記録指標がbit一致した。
[開始確認](experiments/logkv-no-position-main-20260913/launch.json)。
準備commit `c30b207`。全体見積もり約4.1時間、終了は9月14日2時台が目安。
事前GPU検証を含む停止期限は2026-09-14 05:43:53 JST。

追加依頼により、同条件の[3層比較](logkv-no-position-3layer.md)を23:40 JSTからGPU2/3で開始した。
この2層実験はGPU0/1で継続し、元の実行設定・停止期限を維持する。

## 既存データとの関係

この条件は未測定ではなく、[2026-09-07の相対位置補正比較](logkv-relative-position-experiments.md)の
`none`（phase_emb=False、relative_position_bias=False）として両タスクを測定済みだった。
今回の依頼では、mainをそのまま使う再確認として、この1構成・2タスクだけを新規学習する。

過去のnoneの完全一致数（各256例、best / final）：

| T | Copying | Selective Copying |
|---:|---:|---:|
| 16 | 205 / 156 | 9 / 4 |
| 64 | 26 / 17 | 1 / 0 |
| 131,072 | 1 / 1 | 0 / 0 |

元の設定・結果とSHA256は[過去データの参照記録](experiments/logkv-no-position-main-20260913/historical_reference.json)に保存。
過去のnoneは実装`7261652`上で相対補正を無効にしたもので、厳密な決定論モードを使っていなかった。
今回は決定論モードを有効にするため、過去runの数値と完全に一致することは前提としない。
同じ学習seedの再実行であり、独立seedを増やす実験ではない。

## 今回の条件

凍結ソースはmain commit `0932d8c`。使用するモデル・通常訓練コード・タスク・依存設定の
全27ファイルが、重複除去commit `3b0ce51`とGit上で一致することを確認した。
凍結先とファイルSHA256は[ソース記録](experiments/logkv-no-position-main-20260913/source_manifest.json)を参照。

| 項目 | 設定 |
|---|---|
| アーキテクチャ | 重複なしLogKV、C4、d512、8 heads、d_ff1024、2層 |
| 位置埋め込み | 無効。RoPE・相対位置補正などの追加位置機能もなし |
| 維持する設定 | gated attention、self slot、固定−i log C、KV/V normなし |
| パラメータ数 | 5,786,112 |
| タスク | 固定10桁、Copying / Selective Copyingを別モデルとして初期状態から学習 |
| 学習 | 各50,000 steps、batch64、grad accumulation1、T loguniform[1,2028]、全位置CE |
| 最適化 | AdamW、lr3e−4、warmup1000、weight decay0、clip1 |
| seed | 初期化0、訓練データ1、評価12345 |
| 精度 | 学習fp32重み＋bf16 autocast、評価bf16重み＋autocast |
| 評価 | 各タスクbest/final、T131072まで標準41点、各256例、計164セル |

`phase_levels=2`は比較時のメタデータとして揃えるが、`phase_emb=False`なので埋め込みは生成しない。
階層やcausal maskに由来する位置的な構造は残っており、完全な順序不変モデルという意味ではない。
bestは訓練100 steps区間のstring/token/−EMA lossの辞書順で選択し、評価値で選ばない。
2タスクのデータ乱数消費順は異なるため、タスク間で各stepのT系列が同一とはしない。

両タスクは保存した共通初期重みとの全テンソル一致を本学習開始時にも検査する。
`torch.use_deterministic_algorithms(True)`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、
cuDNN benchmark=False。短い20 stepsを各タスクでGPU0/1にて反復し、重みと記録指標がbit一致した。
この短い反復は50k全体や異なる環境でのbit再現性を保証しない。

## 実行時間と検証

同じモデルサイズ・batch・T分布での300 stepsの同時実測は、Copying 57.9秒、Selective 54.8秒。
50kへの単純外挿は2.681時間 / 2.537時間。
15%の学習時間余裕、best/final評価1時間、GPU事前検証を含む全体見積もりは**4.11時間**。
[事前検証記録](experiments/logkv-no-position-main-20260913/preflight.json)。

GPU0=Copying、GPU1=Selective Copying、最大2GPU。
両タスクは独立した訓練・checkpoint・評価として並列に実行する。
事前GPU確認開始から7.5時間で停止し、失敗時には両workerと子プロセスを停止する。
追加seed、phase2再学習、位置方式の比較、16M延長は自動追加しない。

評価では各例の記憶列・実配置・10桁の予測・logit marginを保存する。
完了後はCPUで各タスクの元generatorを再生し、同じminibatch・seedで全164セルの記憶と配置を照合する。
正答数・桁別誤答・50k完遂・best選択・設定・重みハッシュも確認する。
300 stepsの実checkpointから、両タスクbest/finalをT3/16/65・各256例で評価し、
元generatorの記憶・配置と保存予測の正答数の照合を通過した。
[評価器の動作確認](experiments/logkv-no-position-main-20260913/evaluation_smoke.json)。

大型生成物：`/mnt/raid0/RecursiveCompressor/experiments/logkv-no-position-main-20260913/`。
[実行スクリプト](experiments/logkv-no-position-main-20260913/README.md)。
