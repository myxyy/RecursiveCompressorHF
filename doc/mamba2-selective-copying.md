# Mamba-2による固定10桁Selective Copying比較

2026-09-28、ユーザー依頼により、前回のCopyingと同じMamba-2を使って
Selective Copyingを新規学習・評価した。**9月28日00:51:32 JSTに50,000-step学習・best/final全82セル評価・CPU解析が完了。GPU0は解放済み。**
00:04:46 JST開始、学習・評価46.7分、事前測定を含め約50分。
見積もり約1.47時間・停止期限07:31:26 JST以内に完了した。
[開始記録](experiments/mamba2-selective-copying-20260928/launch.json)、
[完了後の再確認](experiments/mamba2-selective-copying-20260928/completion-recheck.json)。

元の[Mamba論文](https://arxiv.org/abs/2312.00752)は内容に応じた選択を検証するために
Selective Copyingを扱っている。今回はMamba-2・本リポジトリのタスク条件での比較であり、
論文の数値の直接再現ではない。[公式実装との対応・依存環境](mamba2-copying.md)は前回と同じ。

## 完了結果

**Mamba-2は訓練範囲付近のSelective Copyingを高精度に解けたが、大幅な距離外挿では性能が低下した。**
bestは46,800 step、finalは50,000 step。bestは訓練指標で選択しており、評価距離別には選び直していない。
bestは評価したT≤3,072の全30点で256/256完全一致、finalはT≤2,048の全29点で256/256だった。
LogKVの対照は既存の標準構成（CausalConv4・位相埋め込みなし・固定減衰）で、今回再学習していない。

下表の完全一致数は各256例。best / finalの順で、両モデルに同じ評価例を使用した。

| T | Mamba-2 完全一致 | LogKV 完全一致 | Mamba-2 桁精度 | LogKV 桁精度 |
|---:|---:|---:|---:|---:|
| 64 | 256 / 256 | 94 / 112 | 100% / 100% | 90.39% / 90.82% |
| 256 | 256 / 256 | 29 / 35 | 100% / 100% | 81.25% / 82.62% |
| 1,024 | 256 / 256 | 10 / 19 | 100% / 100% | 77.15% / 78.55% |
| 2,048 | 256 / 256 | 14 / 30 | 100% / 100% | 77.50% / 80.55% |
| 3,072 | 256 / 246 | 10 / 29 | 100% / 99.61% | 74.88% / 80.23% |
| 4,096 | 240 / 216 | 4 / 11 | 99.38% / 98.28% | 71.76% / 75.08% |
| 6,144 | 107 / 61 | 11 / 19 | 88.95% / 86.48% | 74.49% / 78.40% |
| 8,192 | 26 / 19 | 3 / 11 | 76.02% / 76.72% | 68.40% / 71.48% |
| 12,288 | 1 / 2 | 0 / 11 | 56.13% / 56.72% | 60.78% / 68.83% |
| 16,384 | 0 / 0 | 0 / 3 | 44.38% / 38.75% | 57.77% / 65.12% |
| 131,072 | 0 / 0 | 0 / 0 | 15.08% / 13.28% | 41.05% / 48.24% |

![Mamba-2 and LogKV Selective Copying comparison](experiments/mamba2-selective-copying-20260928/results/comparison.png)

訓練100-step区間の完全一致率が初めて100%に達したのは14,800 step。
最終区間も完全一致率・桁精度とも100%、EMA lossは8.48e-7だった。
評価T=2,048は訓練上限2,028をわずかに超えているが、両checkpointで全例完全一致した。
したがって、今回の設定では、訓練範囲付近の選択・順序保持能力と、大幅な距離外挿の能力を分けて評価する必要がある。

### 結果からの考察

- **選択して10桁を覚える能力は、今回のMamba-2が標準LogKVより明確に高い。**
  T=1,024で256/256対10/256（best）であり、LogKVよりパラメータ数が少なくても高精度だった。
  内容に応じて状態の更新・保持を変える仕組みが有効という解釈と整合するが、
  内部のゲートを操作する検証はしていないため、成功機序を特定したわけではない。
- **長距離外挿は別の課題として残る。** Mamba-2はbest/finalともT≥16,384の評価点で完全一致0/256。
  T=131,072の桁精度は15.08% / 13.28%で、数字1..8を一様に推測する12.5%に近づく。
  同TでLogKVも完全一致は0/256だが、桁精度41.05% / 48.24%は残った。
  これは入力を選択する能力の不足だけでは説明できず、長い空白区間での状態保持や外挿を調べる動機になる。
- **誤りは最後の桁から一様に増えるわけではない。** T=8,192のbestでは第7桁が154/256誤答なのに対し、
  第10桁は全例正解。finalも第7桁が186/256誤答で、第9・10桁は各1/256誤答。
  T=131,072では予測列の種類がbestで34、finalで9まで減り、最頻列に83例 / 93例が集中した。
  前回のCopyingのように全例が一つの回答に固定したわけではないが、出力の多様性が大きく失われている。

同じ小規模Mamba-2の[Copying結果](mamba2-copying.md)と合わせると、
「訓練範囲付近のSelective CopyingはMamba-2が強く、極端な距離での固定10桁CopyingはLogKVが強い」
という違いが今回の条件では見られた。単一seed・異なるパラメータ数・BF16評価の比較であり、
Mamba系列全体の限界や、LogKVのSelective Copying改善が不可能という結論ではない。
FP32評価や空白中の状態変化・ゲートの計測は追加診断候補だが、今回は実施していない。

### 監査と成果物

終了後にCPU解析を再実行し、全82セルの予測・桁別誤答・margin、best選択、重み・凍結ソース・上流コードの
ハッシュを再確認した。正解列はseedから再生成し、best/finalとLogKV対照に一致した。
配置位置・記憶数字についても事前に全41点×256例でLogKVの保存記録との一致を確認済み。
既存LogKVの82セルも予測から正答数を再集計した。今回の再確認でGPUは追加使用していない。

事前測定と本学習の最初300 stepsはbit一致しなかった。GPU演算は決定論モードに固定しておらず、
同じseed・設定であっても学習軌跡のbit再現性は保証しない。
状態の保存量は全評価点で1例あたり1,063,936 bytesと一定だった。

- [全82セル集計・桁別誤答](experiments/mamba2-selective-copying-20260928/results/metrics.json)
- [比較グラフ](experiments/mamba2-selective-copying-20260928/results/comparison.png)
- [bestの全予測・margin](experiments/mamba2-selective-copying-20260928/results/best.json.gz) / [final](experiments/mamba2-selective-copying-20260928/results/final.json.gz)
- [CPU監査結果](experiments/mamba2-selective-copying-20260928/results/review.json) / [解析コード](experiments/mamba2-selective-copying-20260928/analyze_completed.py)
- [学習ログ](experiments/mamba2-selective-copying-20260928/train_log.jsonl) / [完了状態](experiments/mamba2-selective-copying-20260928/supervisor.json)

## 設定

- 固定10桁、数字1..8を先頭T+9位置にランダム配置、末尾11個のmarkerに続く出力で順序通り再生。
  入力系列長はT+20。既存`exp.selective_copying.task`をそのまま使用する。
- Mamba-2：2層、d_model512、d_state128、expand2、headdim64、ngroups1、d_conv4、
  scan_chunk_size256、3,445,856パラメータ。重み共有・初期化・非融合経路は前回と同じ。
- 新規初期化seed0、50,000 steps、batch64、AdamW lr3e-4、warmup1,000、weight_decay0、
  clip1、T=1..2,028のloguniform、全位置の位置合わせCE。Copying重みからの転移はしない。
- bestは訓練100-step区間の完全一致率・桁精度・EMA lossで選択し、finalとともに評価する。
  訓練中の追加quick evaluationは無効。学習はFP32 master重み＋BF16 autocast。
- 評価：T=1..14と2の累乗・中間点、最大131,072の41点×各256例×best/final、計82セル。
  seed12345、BF16重み＋autocast、入力はCPU生成、8,192トークンずつ状態を引き継ぐ。
- 比較対象は[標準LogKV＋CausalConv](logkv-causal-conv.md)のSelective Copying best/final。
  LogKVは5,792,256パラメータであり、同パラメータ数の比較ではない。学習seedは各1個。

## 事前確認と実行

事前にT=131,072まで全41点×8例の評価器smokeを通過し、状態量は1例あたり1,063,936 bytesで一定だった。
また、全41点×256例の数字・配置位置・正解列が既存LogKVの保存済みデータと一致した。
[タスク整合性検証](experiments/mamba2-selective-copying-20260928/task-validation.json)。

300-step実測は100→300 stepsで0.053秒/step。
訓練30%の余裕と評価30分を含む見積もりは約1.47時間。
GPU0の1台のみ使用し、事前測定開始から7.5時間で停止する。
この1学習とbest/final評価・CPU監査までで終了し、追加タスク・別seedは自動起動しない。

本実験では既存のMamba-2キャンペーンと評価器に`--task selective-copying`を追加した。
省略時は従来どおりCopying。モデル本体・既存Copying実験成果物は変更していない。

```bash
# 単独の訓練
uv run --extra mamba2 python -m exp.selective_copying.train --arch mamba2 \
  --run-name mamba2-selective --d-model 512 --num-layers 2 \
  --max-t 2028 --t-dist loguniform --steps 50000 --batch-size 64 \
  --lr 3e-4 --warmup 1000 --seed 0 --eval-interval 0

# 予測・marginを保存する評価（best/finalそれぞれに実行）
uv run --extra mamba2 python -m exp.mamba2_copying.evaluate \
  --task selective-copying --model-dir /path/to/model_best \
  --output /path/to/results/best.json
```

大型成果物・凍結ソース・ログ：
`/mnt/raid0/RecursiveCompressor/experiments/mamba2-selective-copying-20260928-run/`。
`campaign.json`が進行状態、`train.log`が学習ログ、`results/{best,final}.json`が全予測。
本学習成功後、CPU解析スクリプトが予測・margin・重みハッシュ・凍結ソース・上流コード・
best選択を監査し、LogKVとの比較図と集計を保存する。
[事前確認・実行記録](experiments/mamba2-selective-copying-20260928/)。
