# 階層共有の位置表現：可変桁数Copying比較

実験ブランチ `logkv-position-study`（分岐元 `a6c5a87`、実装commit `24b360c`）。
2026-09-07 16:27〜09-08 05:33（JST）に6 GPUで実施し、全12 runの学習と評価を完了した。
前回の[圧縮logit減衰](logkv-compressor-experiments.md)を受け、
圧縮される内容に子位置を結び付ける方式と、読み出しに相対Key/Valueを使う方式を比較する。
本レポートと過去の全実験記録はmainに収録し、実験用の実装は上記ブランチに置く。

**Copyingの完全一致という目標は未達。** 相対K/Vや併用時のレベル減衰除去には一部の条件で改善があり、
Selectiveでは短いTのM32にも完全一致例が得られた。しかし、記憶桁数・待ち時間を広げた全条件で
安定して復元できる方式は得られなかった。以下にbest/finalの差を含む全結果を記録する。

## 実装

`--compressor-position-transform`では、C個の子の内容から従来どおりattention重みaを計算し、
`k_parent = Σ_c a_c U_c k_c`、`v_parent = Σ_c a_c U_c v_c`とする。q_parentは末尾q。
各Uは固定のチャネル置換と2個の学習可能なHouseholder反射の合成。
`H(u)x = x - 2 û (ûᵀx)`、`û = u/||u||`。
子位置・head・Transformer層ごとに別のUを持ち、K/Vと全圧縮階層では共有する。
反射ベクトルのノルムが正規化eps以上なら、変換は実数演算では直交で、有限精度では丸め誤差を伴う。
反射は最低fp32で計算。ゼロ近傍はノルムをeps=1e-12で下限処理するため、
その領域では厳密なHouseholder反射とは限らない。学習後のノルムと直交性も保存して確認する。
追加学習パラメータは2層×8 heads×C4×2反射×head_dim64=8,192個。
置換はcheckpointに保存され、学習されない。

`--relative-position-kv`では、レベル内のQuery子番号jと完成済み子番号cについてd=j-cを使う。
`logit = qᵀ(k_c+r^K_d)/sqrt(d_head)`、`value = v_c+r^V_d`。
既存の全階層共通softmaxを維持する。self slotには距離0のrを使う。
C種類のベクトルをhead・Transformer層別に持ち、全圧縮階層で共有する。
追加学習パラメータは8,192個。Shawらの相対Key/Value表現を階層内の距離に適用した案であり、
[原論文](https://aclanthology.org/N18-2074/)がこの圧縮モデルでの性能を保証するものではない。

`--level-decay-scale 0`で固定の `-i log C` を無効にできる。
デフォルトはscale=1、両方の新位置表現は無効で、既存checkpointの設定を維持する。
設定を変えた場合は旧hiddenを使わず、hidden=Noneから開始する。

## 学習前の診断

少数の反射だけを共有する初案では、各headで高々2C本の反射ベクトルが張る共通部分空間だけが
位置によって変化し、その直交補空間はどの子位置でも不変だった。階層を増やしても
この部分空間は広がらず、合成診断の64桁×8記号の符号行列はrank72に留まった。
固定の子位置別チャネル置換を加えるとrank512となり、fp64の擬似逆行列による線形復号では
4・16・64桁の各256例で完全一致した。内容logitをすべて0にした合成条件で、LMの性能ではない。
圧縮後の要約だけをbf16に丸めた場合も各256例で100%だった。ただし、これは
再圧縮の各段階でbf16の丸め誤差が蓄積する条件を検証したものではない。
前回衝突した4進位置01/10の交換にも非ゼロの出力差が生じた。
relative Valueは、Valueが同一の場合にも読み出し位置による出力差を作った。
位置表現なしとrelative Key/Valueのみの圧縮は、合成診断では順序を復号できない。

固定置換を加えた符号行列の診断値は以下。特徴数はM×8記号。

| M | rank / 特徴数 | 条件数 | fp64復号の完全一致 | 最終要約をbf16化した復号 |
|---:|---:|---:|---:|---:|
| 4 | 32 / 32 | 1.56 | 100% | 100% |
| 16 | 128 / 128 | 2.90 | 100% | 100% |
| 64 | 512 / 512 | 1,284.93 | 100% | 100% |

独立した圧縮・読み出し参照とのfp64出力/入力勾配/パラメータ勾配一致、C=2/3/4、
分割逆伝播、逐次推論、因果性、head/chunk対応、保存読込、bf16推論、可変長タスク、
回答が推論チャンクをまたぐ採点を検証：265 tests passed。

## 比較条件（本学習前に固定）

| mode | 圧縮位置変換 | 相対K/V | 位相埋め込み | 固定レベル減衰 |
|---|---|---|---|---|
| none | なし | なし | なし | -i log C |
| phase2 | なし | なし | 2階層・周期16 | -i log C |
| binding | あり | なし | なし | -i log C |
| relative-kv | なし | あり | なし | -i log C |
| combined | あり | あり | なし | -i log C |
| combined-no-decay | あり | あり | なし | なし |

全条件で前回のスカラー相対logit減衰とCompressor logit減衰は無効。
重複なし構造、gated attention、self slot、KV/V normなし。
6方式×Copying/Selective Copyingを各50,000 steps学習した。
6 GPUに1方式ずつ割り当て、各GPUでCopying、Selectiveの順に実行した。
各runは共通の初期重みから独立に学習し、CopyingのcheckpointをSelectiveに引き継がない。

- d512 / H8 / ff1024 / 2層 / C4、AdamW lr3e-4、warmup1000、weight decay0、clip1
- batch64、microbatch32×2回蓄積、fp32 master weights＋bf16 autocast
- Mは{10,16,32,64}から一様選択、待ち時間Tはloguniform[1,2028]、blank prefix Pは整数[0,63]から一様選択
- 1 stepのmicrobatch間でM/T/Pを共有。入力長はP+T+2M、最大2,219。
  記憶記号は1〜8、blank=0、末尾はM+1個のmarker=9、最後のM位置が回答。
  正解トークンを次の入力に戻す形式ではなく、全位置のposition-aligned CEで学習
- Selectiveはprefix後のT+M−1位置からM箇所を選び、出現順にコピーする
- 学習seed0、内容生成seed1、M/T/P生成seed2。各方式の共通重みはphase2も含めて明示的に同一値をロード
- 各条件1 seed。GPUの速度差と厳密な決定論モードを無効にした実行を含むため、所要時間を方式の速度比較に使わない

前回の固定M10・prefix0とは学習問題が異なるため、過去の数値を新条件の対照として流用しない。
新規phase2も含めて再学習する。M64までを学習する比較は「記憶桁数の範囲を広げた問題」であり、
M64への未学習長外挿を意味しない。

bestは2,000 stepsごとに独立した検証データで選ぶ。
検証はM4種×T{16,64,256,1024}×P{0,7}、各32例、seed54321。
セルごとの完全一致率の平均、token正解率の平均、−訓練EMA lossの辞書順。
検証集合はcheckpoint選択に使うため、最終評価集合とは分離する。

## 最終評価

各runのbest/finalをfp32重み＋bf16 autocast、seed12345、各セル256例で評価した。
データ生成seedはtask/M/T/Pから決定するため、評価順序によって入力が変わらない。

- M={10,16,32,64}×T=1〜131,072の従来と同じ41点、P=0：164セル
- M4種×T{16,64,256,2048}×P{7,15,63}：48セル
- 未学習M128×T{16,64,256,2048}×P{0,15}：8セル

合計220セル×2 checkpoint×12 run=5,280評価行。完全一致率とtoken正解率を両方記録した。
全runの終了コード、共通初期重み・設定・学習Mの出現数、best選択、評価セルと正解数の整合性を検証した。
アーカイブの生ログ・設定84ファイルと実行manifest12件は元ファイルとバイト単位で一致した。
短い検証・評価prefixは学習分布内であり、prefix外挿の実験ではない。
今回は候補1/2の分離比較を優先し、K/Vの相対回転による座標系の再基準化案は含めない。

選ばれたbestのstepは以下。finalはすべて50,000 step。

| task | none | phase2 | binding | relative-kv | combined | combined-no-decay |
|---|---:|---:|---:|---:|---:|---:|
| Copying | 48,000 | 48,000 | 48,000 | 48,000 | 46,000 | 50,000 |
| Selective | 48,000 | 50,000 | 50,000 | 50,000 | 48,000 | 44,000 |

## Copyingの結果

6方式すべて50,000 stepsの学習とbest/final各220セルの評価を完了した。
**今回の学習条件では、新方式による完全一致の目標は達成できなかった。**
M=32・64および未学習M=128では、prefix評価も含む全方式・両checkpointの全セルで
string_acc=0だった。これは各セル256例で完全一致した例がなかったという意味であり、
token正解率まで0だったわけではない。

以下はP=0の完全一致率（%、各セル256例）。`best / final`の順。
右列はM=10についてTの41点を等重み平均した値で、学習分布に従った期待精度ではない。

| mode | M10・T64 | M16・T64 | M10・全41 horizon平均 |
|---|---:|---:|---:|
| none | 0.00 / 0.00 | 0.00 / 0.00 | 0.03 / 0.05 |
| phase2 | 99.22 / 97.27 | 1.95 / 3.52 | 79.66 / 67.08 |
| binding | 0.00 / 0.00 | 0.00 / 0.00 | 0.06 / 0.02 |
| relative-kv | 23.44 / 16.41 | 0.00 / 0.00 | 17.70 / 11.05 |
| combined | 0.00 / 0.00 | 0.00 / 0.00 | 0.14 / 0.24 |
| combined-no-decay | 75.39 / 75.39 | 2.34 / 2.34 | 48.12 / 48.12 |

相対K/Vのみの方式はM10で位置表現なしを上回ったが、phase2には届かなかった。
圧縮位置変換のみ、または固定レベル減衰を残した併用では、M10でも完全一致率は低かった。
併用からレベル減衰を外すとM10の精度は大きく改善したものの、長い待ち時間に対して安定せず、
T2048では4.30%、T131072では0%だった。M16・T64も2.34%に留まった。

phase2もM10・T131072ではbest 0.39%、final 0%まで低下した。
M10・T2048はbest 96.88%に対しfinal 42.58%であり、最終stepだけでもbestだけでも
学習の安定性を十分には表せない。全方式・両checkpointを通じ、100%だったのはphase2 bestの1セルのみで、
新方式には100%のセルがなかった。
combined-no-decayのbestは50,000 stepで、best/finalの重みと全セルの正解数が同一。
独立した2回の再現実験としては数えない。

prefixを変えたM10・T64のbest完全一致率は、P=0/7/15/63の順に
phase2が99.22/94.92/96.48/94.14%、combined-no-decayが75.39/76.17/75.00/72.27%。
開始位置を多少ずらしてもM10の成績は保たれたが、学習済みprefix範囲内の結果である。

![Copying bestの完全一致率](experiments/logkv-position-study-20260907/copying-best-string_acc.png)

縦の点線は学習時のT上限2028。横軸は待ち時間Tで、記憶桁数Mは各パネルに示す。
[finalの完全一致率](experiments/logkv-position-study-20260907/copying-final-string_acc.png)、
token正解率の[best](experiments/logkv-position-study-20260907/copying-best-token_acc.png) /
[final](experiments/logkv-position-study-20260907/copying-final-token_acc.png)も保存した。

## Selective Copyingの結果

全6方式の学習とbest/final評価を完了した。全方式・両checkpointを通じ、100%のセルはなかった。
以下はP0の完全一致率（%、`best / final`）。右列はM10の41 horizon等重み平均。

| mode | M10・T64 | M16・T64 | M10・全41 horizon平均 |
|---|---:|---:|---:|
| none | 0.00 / 0.39 | 0.00 / 0.00 | 0.02 / 0.03 |
| phase2 | 4.69 / 4.69 | 0.00 / 0.00 | 11.25 / 11.25 |
| binding | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| relative-kv | 17.97 / 17.97 | 0.00 / 0.00 | 12.80 / 12.80 |
| combined | 0.00 / 0.00 | 0.00 / 0.00 | 0.33 / 0.50 |
| combined-no-decay | 8.98 / 5.08 | 0.00 / 0.00 | 18.14 / 18.75 |

bestのM10・T64・P0では、relative-kvの完全一致率17.97%がphase2の4.69%を上回った。
token正解率も82.97%対73.98%。ただし、同じT64のM16では全方式のbestが完全一致率0%だった。

combined-no-decayは短いTで改善した。best・P0・T1の完全一致率は
M10=71.09%、M16=41.02%、M32=12.50%。M32ではT2/3/4にも15/10/2例の完全一致があった。
combinedのbestもM32・T1で1/256例が完全一致した。
bestにおけるM32の完全一致例はT≤4・P0に限られた。
M64と未学習M128は、全方式・両checkpointの全セルで完全一致率0%だった。

combined-no-decayのfinalでは、T1の完全一致率がさらに上がった。

| M | T1・P0のbest完全一致率 | final完全一致率 |
|---:|---:|---:|
| 10 | 71.09% | 76.56% |
| 16 | 41.02% | 73.83% |
| 32 | 12.50% | 45.31% |
| 64 | 0% | 0% |

M32・finalの完全一致例はT1/2/3/4/6の116/39/19/3/1例に限られ、T64では0%だった。
T1はSelectiveでも記憶区間にblankが入らない条件である。
このcheckpointはSelectiveの学習分布から得られたもので、Copying学習runとは別に扱う。

bestのcombined-no-decayのM10・41 horizon平均完全一致率は18.14%で、relative-kvの12.80%、
phase2の11.25%を上回る。しかし、T64のM10では8.98%とrelative-kvより低く、
T2048では0%になった。方式の順位は待ち時間と記憶桁数に依存しており、
短いTで得た改善を長い待ち時間に維持できたわけではない。

finalでT1が改善しても、M10・T64はbestの8.98%から5.08%へ下がった。
bestはT1を含まない所定の検証集合で選んでおり、すべての評価セルの最良checkpointを意味しない。
phase2・binding・relative-kvはbest/finalの重みと全セルの正解数が同一だった。

開始位置を変えたrelative-kvのM10・T64完全一致率は、P0/7/15/63で17.97/19.14/9.38/19.53%。
prefixによる変動も残った。T131072では全方式・全Mのbest/finalとも完全一致率0%。
M10のtoken正解率はrelative-kvが19.38%、combined-no-decayがbest 13.32% / final 13.16%だった。

![Selective Copying bestの完全一致率](experiments/logkv-position-study-20260907/selective-copying-best-string_acc.png)

[finalの完全一致率](experiments/logkv-position-study-20260907/selective-copying-final-string_acc.png)、
token正解率の[best](experiments/logkv-position-study-20260907/selective-copying-best-token_acc.png) /
[final](experiments/logkv-position-study-20260907/selective-copying-final-token_acc.png)も収録した。

## 学習済み圧縮器の事後診断

Copyingの低い完全一致率を確認した後、学習済みU自体が順序を区別できなくなったかを
切り分けるために追加した診断。学習前に固定した主比較・checkpoint選択には使っていない。
学習済みUを取り出し、学習前と同じ固定乱数の8記号ベクトルと均一な圧縮重みを用いて
符号行列を作り、外付けの擬似逆行列で復号する。各checkpointのSHA256も保存する。
**実際のLMが学習した記号表現、内容依存の圧縮重み、Queryによる読み出しを評価するものではない。**
M個の記号を1個の要約にする診断であり、blankの待ち時間Tに伴う追加の再圧縮も含まない。

各タスクで3方式×best/final×2層×M{4,16,64}を診断した。
計72ケースすべてでfull rank（相対閾値1e−10）を維持し、fp64の完全一致率は各256例で100%だった。
最終要約だけをbf16に丸めた場合は66ケースで100%。例外の6ケースはすべてM64で、以下のとおり。
層番号は0始まり。best/finalが同じ重みの場合も別行として記録しており、独立した反復実験ではない。

| task | mode | checkpoint | 層 | 条件数 | bf16要約の完全一致数 / 256 |
|---|---|---|---:|---:|---:|
| Copying | combined | best | 0 | 15,442.90 | 200 |
| Copying | combined | final | 0 | 8,536.57 | 252 |
| Selective | combined | final | 0 | 8,800.07 | 242 |
| Selective | combined-no-decay | best | 0 | 5,000.59 | 254 |
| Selective | combined-no-decay | final | 0 | 9,048.95 | 239 |
| Selective | combined-no-decay | final | 1 | 13,629.82 | 205 |

したがって、単純な順序衝突はこの合成条件では解消された一方、数値的な復号の余裕には差があり、
それだけでLMのコピー能力が得られるわけではなかった。
各圧縮段階でのbf16丸めを含む実際の再帰計算の頑健性は、この診断では判定できない。

保存された全変換の反射ベクトルのノルムは6.54〜9.82で、正規化eps近傍への崩壊はなかった。
fp64で再構成した直交行列の最大誤差は2.11e−15。これらの統計と相対K/Vのノルムも
checkpoint別の `position_stats_*.json` に保存した。

## 実行と成果物

実行記録の親ディレクトリは
`/mnt/raid0/RecursiveCompressor/experiments/logkv-position-study-20260907`。
実装をcommitした固定worktreeをsourceとして使い、mode/task別に設定・学習ログ・モデル・
best選択記録・検証値・最終評価値・実行コマンドと終了コードを保存する。

以下の学習・評価コマンドと新方式のcheckpoint読込には、実験commit `24b360c` のworktreeを使う。
mainには文書と評価データを統合し、実験用CLIと新位置表現の実装は実験ブランチに置く。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run python exp/position_study/train.py --mode combined --task copying \
  --run-dir /path/to/new-run
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run python exp/position_study/evaluate.py --task copying \
  --run-dir /path/to/new-run --checkpoint best
```

既存runへの上書きは拒否する。両checkpointを評価するにはfinalでも実行する。
optimizerとデータ生成用の乱数状態も定期保存するが、自動resume機能は本runnerには含めていない。

リポジトリにはモデル重みを除く以下の記録を収録する。

- [成果物・再集計手順](experiments/logkv-position-study-20260907/README.md)
- [全5,280評価行のCSV](experiments/logkv-position-study-20260907/comparison.csv)と
  [M別・checkpoint別集計](experiments/logkv-position-study-20260907/summary.json)
- [代表Tのtoken / string正解率表](experiments/logkv-position-study-20260907/tables.md)と
  [prefix・未学習M128の全評価表](experiments/logkv-position-study-20260907/prefix-and-unseen-memory.md)
- [学習済み圧縮器の事後診断表](experiments/logkv-position-study-20260907/trained-encoder-diagnostics.md)
- [学習・検証曲線](experiments/logkv-position-study-20260907/learning.png)

## 結果の解釈上の制約

この比較ではMとprefixを同時に変動させる。各Mの学習回数は約12,500 stepsなので、
固定M10で50,000 steps学習した過去の結果との差だけから、周期16の位相埋め込みの限界を
断定することはできない。方式間の比較には今回の同条件runを使う。

新しい位置パラメータは子位置・局所距離にだけ依存し、圧縮階層を増やしてもパラメータ数や
位置テーブル長を増やす必要はない。一方、チャンク境界は系列先頭を基準に決まるため、
全体をシフトしたときに厳密に不変な表現ではない。prefix評価はその境界依存性も含む。
O(log L)の保持状態と無期限の逐次実行が可能であることと、有限精度で任意長のランダム列を
完全復元できることは別であり、後者を保証する実験ではない。
長いTの評価でも記憶列の開始位置は先頭付近にあり、非常に大きい絶対位置で
新たな記憶列を読み込む場合の性能は未検証である。

レベル減衰の除去は併用条件だけで比較している。その差は併用時の減衰の効果を示すが、
位置表現なし・相対K/Vのみでも減衰を外すと同じ改善になるかは本比較からは分からない。

各セル256例で100%でも、その分布の誤り確率が0と証明されたわけではない。
学習は各条件1 seedなので、方式差と初期値に依存する学習のばらつきを分離していない。
