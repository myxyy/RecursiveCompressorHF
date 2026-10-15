# 固定0レベルバイアス：標準LogKVのCopying / Selective Copying比較

2026-10-14 **13:08:46 JSTに両タスクの学習・best/final評価・自動監査が完了した。**
各50,000 steps、全164セル。GPU0/1は解放済み。
本学習は10:16:47 JST開始、実測2.87時間（事前確認込み2.93時間）。
事前見積もり約4.3時間、停止期限17:42:50 JST以内に完了した。
ブランチ `logkv-no-level-bias`、基点 `ea64f2d`。
[開始確認](experiments/logkv-no-level-bias-20261014/launch.json) /
[完了記録](experiments/logkv-no-level-bias-20261014/campaign.json)。

追加依頼の[Copying 16M評価](experiments/logkv-no-level-bias-20261014/extension-16777216/)も
**14:55:40 JSTに完了。best/finalとも8/8例・80/80桁が完全一致した。**
GPU0/1は解放済み。以下の41点比較に加え、16M追加評価を別節に記す。

## 結果と解釈

**固定0でもCopyingの長距離保持はほぼ維持され、Selective Copyingの中長距離は固定減衰より改善した。**
ただしCopyingの全評価点での完全一致は達成しておらず、Selectiveの長距離完全一致も低い。
同一初期重み・同一評価例の1 seed比較であり、固定0が普遍的に優れるとはまだ判断できない。

### Copying

bestは36,100 step、finalは50,000 step。固定減衰対照はbest/finalとも50,000 step。
全41点×256例の結果は以下のとおり。

| 指標 | 固定減衰 best / final | 固定0 best / final |
|---|---:|---:|
| 全評価例の完全一致 | 10,496 / 10,496 | 10,494 / 10,477 |
| 誤答した桁数（全104,960桁） | 0 / 0 | 2 / 21 |
| T=131,072の完全一致（各256例） | 256 / 256 | 256 / 256 |

固定0のbestはT=5と2,048でそれぞれ255/256、他39点で256/256。
両誤答は第9桁（1始まり）で、直前まで続いた数字をもう一度出力していた。
正解logit minus 最大誤答logitのmarginは−9.1796875と−12.625であり、同点argmaxによる誤答ではない。
finalは19例・21桁を誤り、第5桁16件、第9桁3件、第10桁2件に集中した。
Tとともに単調に記憶が消えるというより、特定の桁・並びの読み出し誤りが残る結果である。
ただし、その発生機構やFP32評価での変化は未検証。

T=131,072の最小正解marginはbest 7.0、final 4.25。
追加依頼により、今回の固定0のbest/finalにもT=16,777,216の評価を行った。
結果は後述の16M追加評価を参照（同じ8例の小規模probeであり、この41点×256例とは別枠）。

[誤答例とmargin](experiments/logkv-no-level-bias-20261014/analysis/copying_errors.json)。

### Selective Copying

固定0のbestは49,000 step、固定減衰対照のbestは48,100 step。両finalは50,000 step。
下表は各256例の完全一致数（best / final）。

| T | 固定減衰 | 固定0 |
|---:|---:|---:|
| 16 | 235 / 239 | 244 / 246 |
| 64 | 94 / 112 | 135 / 121 |
| 256 | 29 / 35 | 75 / 71 |
| 1,024 | 10 / 19 | 44 / 45 |
| 2,048 | 14 / 30 | 65 / 65 |
| 8,192 | 3 / 11 | 49 / 43 |
| 32,768 | 1 / 2 | 23 / 24 |
| 131,072 | 0 / 0 | 18 / 12 |

T≥32の評価点ではbest/finalとも全て固定0の完全一致数が上回った。
短いTでは一部悪化もあるため、全距離での一律改善ではない。
T=131,072の桁精度は、固定減衰 **41.05% / 48.24%** に対し固定0 **78.12% / 75.00%**。
同Tの固定0では第1・2桁はbest/finalとも各256/256正答で、第6〜8桁に誤りが集中する。
bestの桁別誤答数は `[0, 0, 10, 30, 54, 130, 132, 125, 65, 14]`。
記憶の一部はかなり保持できているが、10桁全体の完全一致はbestでも18/256（7.03%）にとどまる。

![固定減衰と固定0の比較](experiments/logkv-no-level-bias-20261014/results/fixed_decay_vs_zero_bias.png)

### この実験から言えること

固定減衰は、この範囲のCopying長距離保持に必須ではなかった。
一方、少数の誤答まで含めたCopyingの完全一致では固定減衰が優れ、
Selectiveの中長距離では固定0が優れるという違いが現れた。

解釈の仮説として、`-i log C` の除去によって圧縮レベルの高い記憶を読み出しやすくなり、
配置が分散するSelectiveの保持に有利になった可能性がある。
ただし今回は再学習したモデルの出力比較であり、attention観測や係数への推論時介入を行っていないため、
直接の機構を確定する結果ではない。標準構成の変更やLLM品質への一般化を結論するには不足する。

10月14日の完了確認では、新旧計328セルの予測を再集計し、8 checkpoint、初期重み、
凍結ソース、実行スクリプト、保存結果のhashとbest選択を再確認した。全て通過。
この41点評価の再監査自体はCPUのみ。その後の追加依頼により、以下の16M評価を実施した。
[全164セル](experiments/logkv-no-level-bias-20261014/results/metrics.json) /
[同一例での比較](experiments/logkv-no-level-bias-20261014/results/baseline_comparison.json) /
[今回の再確認](experiments/logkv-no-level-bias-20261014/analysis/review.json)。

## Copying 16M追加評価（2026-10-14）

ユーザー追加依頼により、固定0のbest（36,100 step）/ final（50,000 step）を
GPU0/1で並列評価した。再学習は行っていない。
14:47:50 JSTに本評価を開始し、14:55:40 JSTに両方完了。実行・終了処理を含め7分50秒。
事前計測の余裕込み約10分の見積もり、30分の停止期限以内に完了し、GPUは解放済み。

| T | best完全一致 | final完全一致 | best最小margin | final最小margin |
|---:|---:|---:|---:|---:|
| 131,072 | 8/8 | 8/8 | 10.40625 | 6.578125 |
| 1,048,576 | 8/8 | 8/8 | 10.3046875 | 6.5625 |
| **16,777,216** | **8/8** | **8/8** | **10.3046875** | **6.546875** |

全セルで80/80桁正答。marginは正解logitから最大誤答logitを引いた値。
T16Mの実評価時間はbest 455.90秒、final 463.55秒。
T1MからT16Mへの延長でも、この8例では正解の余裕をほぼ維持した。

条件は固定10桁・8例、seed12345、batch4、chunk8,192、BF16重み＋BF16 autocast。
Tは待機長であり、総入力長はT+20。
各Tで乱数をリセットし、best/finalと全3距離で同じ8つの列を使用した。
[固定減衰の過去16M評価](experiments/logkv-causal-conv-20260914/extension-16777216/)とも
同じ8列であることをCPUで再確認した（固定減衰も8/8、最小margin10.125）。
この8例は、各Tへ連続して乱数を消費した前節の256例評価から選び出した例ではない。

保存した全logitから予測・margin・正答数を再計算し、乱数seedによる入力再現、
checkpoint/config/凍結ソース/実行スクリプトのhashを監査した。全て通過。
T131,072では標準evaluatorを独立に呼んだ正答数とも一致している。

**固定減衰を外しても、この固定10桁・8例では16Mまで記憶を維持できた。**
ただし前節の41点×256例ではbest 2例、final 19例の誤答があるため、
今回の結果で全入力・全長の完全一致が保証されるわけではない。
Selective Copyingの16M評価や追加学習は行っていない。

[CPU監査・全6セル](experiments/logkv-no-level-bias-20261014/extension-16777216/review.json) /
[bestの予測とlogit](experiments/logkv-no-level-bias-20261014/extension-16777216/best/full.json) /
[finalの予測とlogit](experiments/logkv-no-level-bias-20261014/extension-16777216/final/full.json)。

## 比較する変更

標準の各レベルのlogit加算 `-i * log(C)` を、学習不能な0に固定する。
`--learnable-decay` のゼロ初期化ではなく、係数パラメータを追加しない条件。
自己スロット、CausalConv、圧縮方法、学習データは標準構成のまま。
`--no-level-bias` をCopying / Selective Copying共通トレーナーに追加し、
保存される `LogKVConfig.no_level_bias` を推論・パイプライン読み込みにも反映する。
省略時の動作は従来どおり固定減衰。学習可能減衰・固定増幅との併用は拒否する。

対照は[固定減衰＋CausalConvの固定10桁実験](logkv-causal-conv.md)。
その**未学習の初期重み全てを同一値で再利用**し、両タスクを独立に再学習した。
パラメータ数は対照と同じ5,792,256。
過去のcombined-no-decayは位置表現も異なるため、今回の対照ではない。

## 条件

| 項目 | 設定 |
|---|---|
| モデル | 2層、d512、8heads、FF1024、C4 |
| 位置・局所処理 | 位相埋め込みなし、CausalConv幅4 |
| その他 | gate / self slotあり、KV/V normなし |
| 記憶 | 固定10桁、prefixなし、数字1〜8 |
| 訓練 | 各50,000 steps、batch64、accum1、seed0 |
| 系列 | Tを1〜2,028からloguniform、最大系列長2,048 |
| 最適化 | AdamW、lr3e-4、warmup1,000、weight decay0、grad clip1 |
| 損失・精度 | 全位置aligned CE、FP32 master / BF16 autocast |
| データ乱数 | 訓練CPU generator seed1、簡易評価seed2 |
| checkpoint | 100-step訓練区間の完全一致→桁精度→EMA lossでbestを選択、finalは50k |
| 本評価 | best/final、T=1〜131,072の41点、各256例（全164セル） |
| 評価の精度 | BF16重み＋autocast、seed12345、chunk8,192、token budget 2^19 |
| 並列・時間 | GPU0 Copying / GPU1 Selective、2GPU上限、8時間以上の見積もりなら開始前確認 |

bestは訓練区間だけで選び、評価Tごとの選び直しはしない。
最初の300 stepsの事前benchmarkとの一致、全予測の桁別誤答数、正解logitのmargin、
Selectiveの入力位置を保存・監査した。
評価データはCPUで再生成して照合し、対照の保存済み256例と同一の記憶・配置で比較した。
両タスク完了後に固定減衰との比較表・グラフを自動作成した。

## 検証と再現

CPUテスト228 passed / 6 skipped。FP64独立参照、勾配、分割step/predict、
checkpoint再読み込み、パイプライン出力の一致を含む。
GPUの最大系列長学習・複数精度でのstreaming一致・20-step別GPU再実行・
300-step benchmark・実checkpoint評価smokeは全て通過した。
300 stepsの実測はCopying 62.0秒、Selective 58.5秒。
最大系列長のピーク割当は15.85 GiB、20-stepの別GPU再実行は両タスクの重み・指標がbit一致。
初期重みのSHA256は固定減衰対照と完全一致した。

[実行コード・事前検証](experiments/logkv-no-level-bias-20261014/)。
大きな重みと凍結ソースは
`/mnt/raid0/RecursiveCompressor/experiments/logkv-no-level-bias-20261014/`。
`source_manifest.json` と `implementation.patch` に未コミット変更を含む実行ソースを記録した。
進行中は凍結ソースと実験スクリプトを変更しない。

`run.py` は2タスクの学習→best/final評価→CPU監査を実施する。
事前GPU確認の開始から7.5時間で停止し、失敗時は両workerを停止する。
この一連の実験後に追加seed・16M評価・LM訓練を自動で開始しない。

通常の共通トレーナーでは次のように設定できる（本比較は別途同一初期重みを読ませる）：

```bash
uv run python -m exp.copying.train --arch logkv --no-level-bias \
  --run-name no-level-bias --t-dist loguniform --phase-levels 2
uv run python -m exp.selective_copying.train --arch logkv --no-level-bias \
  --run-name no-level-bias --t-dist loguniform --phase-levels 2
```

1 seedの小規模タスク比較であり、LLM生成品質への効果はこの実験だけでは判断できない。
