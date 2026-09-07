# 階層共有の位置表現：可変桁数Copying比較

実験ブランチ `logkv-position-study`（分岐元 `a6c5a87`）。2026-09-07開始。
前回の[圧縮logit減衰](logkv-compressor-experiments.md)を受け、
圧縮される内容に子位置を結び付ける方式と、読み出しに相対Key/Valueを使う方式を比較する。
本レポートと過去の全実験記録は、実験コードの採否と分けてmainに統合する。

## 実装

`--compressor-position-transform`では、C個の子の内容から従来どおりattention重みaを計算し、
`k_parent = Σ_c a_c U_c k_c`、`v_parent = Σ_c a_c U_c v_c`とする。q_parentは末尾q。
各Uは固定のチャネル置換と2個の学習可能なHouseholder反射の合成。
`H(u)x = x - 2 û (ûᵀx)`、`û = u/||u||`。
子位置・head・Transformer層ごとに別のUを持ち、K/Vと全圧縮階層では共有する。
変換は実数演算では直交で、有限精度では丸め誤差を伴う。反射は最低fp32で計算。
ゼロ近傍のベクトルは正規化のepsで保護する。
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

少数の反射だけを共有する初案では、位置変換が反射ベクトルの張る部分空間内に閉じた。
合成診断の64桁×8記号の符号行列はrank72に留まった。
固定の子位置別チャネル置換を加えるとrank512となり、fp64の擬似逆行列による線形復号では
4・16・64桁の各256例で完全一致した。内容logitをすべて0にした合成条件で、LMの性能ではない。
圧縮後の要約だけをbf16に丸めた復号も診断し、数値精度の影響を別に記録する。
前回衝突した4進位置01/10の交換にも非ゼロの出力差が生じた。
relative Valueは、Valueが同一の場合にも読み出し位置による出力差を作った。
位置表現なしとrelative Key/Valueのみの圧縮は、合成診断では順序を復号できない。

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
6方式×Copying/Selective Copyingを各50,000 steps学習する。
6 GPUに1方式ずつ割り当て、各GPUでCopying、Selectiveの順に実行する。

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

各runのbest/finalをfp32重み＋bf16 autocast、seed12345、各セル256例で評価する。
データ生成seedはtask/M/T/Pから決定するため、評価順序によって入力が変わらない。

- M={10,16,32,64}×T=1〜131,072の従来と同じ41点、P=0：164セル
- M4種×T{16,64,256,2048}×P{7,15,63}：48セル
- 未学習M128×T{16,64,256,2048}×P{0,15}：8セル

合計220セル×2 checkpoint×12 run=5,280評価行。完全一致率とtoken正解率を両方記録する。
長さ別曲線と桁数別表を作り、完全一致に達しない結果やbest/finalの差も含めて報告する。
短い検証・評価prefixは学習分布内であり、prefix外挿の実験ではない。
今回は候補1/2の分離比較を優先し、K/Vの相対回転による座標系の再基準化案は含めない。

## 実行と成果物

実行記録の親ディレクトリは
`/mnt/raid0/RecursiveCompressor/experiments/logkv-position-study-20260907`。
実装をcommitした固定worktreeをsourceとして使い、mode/task別に設定・学習ログ・モデル・
best選択記録・検証値・最終評価値・実行コマンドと終了コードを保存する。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run python exp/position_study/train.py --mode combined --task copying \
  --run-dir /path/to/new-run
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run python exp/position_study/evaluate.py --task copying \
  --run-dir /path/to/new-run --checkpoint best
```

既存runへの上書きは拒否する。両checkpointを評価するにはfinalでも実行する。
optimizerと乱数状態も定期保存するが、自動resume機能は本runnerには含めていない。
