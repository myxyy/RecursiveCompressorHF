# Compressorの位置減衰による記憶タスク評価

実装commit `77356e6`（2026-09-07）。数式・APIは[位置表現実験の設計 §2.5](logkv-position-design.md#25-compressorの位置減衰)。
前回の[retrieval側だけの相対位置補正](logkv-relative-position-experiments.md)の6条件を
すべて評価した後、新規学習を開始した。

## 結果

**Compressorの位置減衰は実装・動作検証できたが、今回の3方式では位相埋め込みの代替には至らなかった。**
学習可能版はCopyingの一部の長さで改善したものの、長さとチェックポイントへの依存が大きい。
Selective Copyingもphase2を下回った。標準構成の位相埋め込みを維持し、圧縮減衰のデフォルトは0のままとする。

6 runすべて50,000 stepsを完了し、best/final各41点×256例を評価した。
2026-09-07 12:41〜15:44 JSTに実行。全18コマンド（学習6＋評価12）が正常終了。
下表はすべて%で、各セルは **best / final**。phase2とretrievalは前回の同条件評価を再掲する。

### T=64の完全一致率

| 条件 | Copying | Selective Copying |
|---|---:|---:|
| phase2（前回） | 100.00 / 100.00 | 17.58 / 16.41 |
| retrievalのみ（前回） | 3.52 / 0.78 | 0.00 / 0.00 |
| half（固定・弱） | 0.00 / 0.00 | 0.00 / 0.39 |
| base（固定・基準） | 0.00 / 0.39 | 0.00 / 0.39 |
| learned（学習可能） | 41.02 / 13.28 | 0.00 / 0.39 |

Copyingのlearnedはretrievalのみの3.52% / 0.78%から41.02% / 13.28%へ改善した。
ただしbestのT=16では49.22%から7.42%、T=256では7.03%から1.17%へ低下しており、
長さ全域の改善ではない。訓練区間の完全一致率もretrievalのみを一貫して上回ってはいない。
Selectiveのfinalの0.39%は256例中1例に相当し、大きな改善とは解釈しない。

### T=131,072のtoken正解率

| 条件 | Copying | Selective Copying |
|---|---:|---:|
| phase2（前回） | 100.00 / 100.00 | 25.55 / 26.48 |
| retrievalのみ（前回） | 50.00 / 52.54 | 18.24 / 18.75 |
| half（固定・弱） | 52.81 / 57.19 | 16.72 / 16.29 |
| base（固定・基準） | 54.34 / 65.59 | 15.66 / 15.98 |
| learned（学習可能） | 57.27 / 41.17 | 16.25 / 16.33 |

この長さのCopyingの完全一致率は、phase2がbest/finalとも100%、baseのfinalが0.39%、
それ以外の圧縮減衰条件は0%。phase2は全41点でtoken・完全一致率100%を保った。
Selectiveの完全一致率はこの長さでは全条件0%。
learnedのCopyingはbestからfinalに長距離token正解率が57.27%→41.17%へ低下した。
一方baseは54.34%→65.59%なので、学習可能版が固定版より安定して良いという結論にはならない。

[代表長さの全表](experiments/logkv-compressor-20260907/tables.md)と
[全820行の比較CSV](experiments/logkv-compressor-20260907/comparison.csv)を保存した。
820行のうち492行は新規6 run、328行は前回phase2/retrievalの4 run。

![bestの長さ別精度](experiments/logkv-compressor-20260907/comparison_best.png)

![finalの長さ別精度](experiments/logkv-compressor-20260907/comparison_final.png)

### 学習と係数

新規runのbest stepはCopyingがhalf=49,600、base=49,400、learned=49,100、
Selectiveがhalf=43,700、base=44,900、learned=47,700。
これは訓練区間の指標で選んだcheckpointであり、すべての評価長さで最良とは限らない。

![学習曲線](experiments/logkv-compressor-20260907/learning.png)

learnedの最終係数の範囲は次のとおり。両層とも初期範囲は0.115525〜0.462098。

| タスク | Transformer層1 | Transformer層2 |
|---|---:|---:|
| copying | 0.038054〜0.605209 | 0.099872〜0.561851 |
| selective-copying | 0.051529〜0.697420 | 0.115985〜0.673098 |

係数はヘッド・層・タスクごとに異なる値へ更新され、単一の固定値には揃わなかった。
ただし、この結果だけで最適係数や学習可能化の優位性は決められない。
learnedには初期ヘッド分布の違いもあり、各条件1 seedで、LM品質は未評価である。

![学習可能係数の推移](experiments/logkv-compressor-20260907/slopes.png)

## 変更点と検証

```text
compression_logit[c] -= α_h * (C-1-c)
```

Queryは従来どおりチャンク末尾で、C=4の古い順の距離は`[3,2,1,0]`。
KとVを補正後の同じsoftmax重みで圧縮する。q_outは末尾qのまま。
retrieval側の相対項 `-(j-c-1)*log(C)/(C-1)`、固定レベル減衰 `-i*log(C)`、
重複なし参照、gated attention、self slotを維持し、位相埋め込みは無効にする。

- `compressor_decay=0.0` と `learnable_compressor_decay=False` がデフォルト。
  追加パラメータはなく、旧コードとのfp64出力・入力勾配・パラメータ勾配のビット一致を確認
- 固定版は全ヘッド・全圧縮階層で同じαを使用
- 学習可能版はTransformer層・ヘッドごとの `α_h=softplus(raw_decay_h)`。
  圧縮階層間で共有するため、深い階層のための追加パラメータは不要。2層×8 headsで16個
- 独立した圧縮・retrieval参照とのfp64出力・勾配一致、C=2/3/4、ヘッドとチャンクの対応、
  任意分割・逐次推論、因果性、再計算、bf16、保存読込を含む **221 tests passed**
- GPUでbatch64・系列長2,048のforward/backward/AdamWを2回実施。
  固定版・学習可能版とも有限なlossと勾配、学習係数の更新を確認。
  ピークallocatedメモリはどちらも約14.81 GiB

末尾以外のK/Vスロットの順序を入れ替えたとき、減衰なしでは同じ圧縮結果だが、
減衰ありでは異なることを確認した。ただし、全valueが同一なら正規化加重平均も同一となる
性質は残る。これは同一トークンの連続を数える能力を保証する仕組みではない。

補助診断では、内容logitをすべて0にして位置ごとに異なるvalueを入力した。
C=4の1回の圧縮では、位置1と2の交換による出力差は固定基準版で約0.102だった
（減衰なしでは0）。一方、16要素を2回圧縮した場合、位置1と4の交換による差は
固定基準版・学習可能版の初期状態とも0だった。
内容logitが等しいとき、深さnの木で元valueに掛かる重みは
`exp(-α_h * Σ_i distance_i) / Z_h^n`となり、同じ距離の総和を持つ経路は同じ重みになるため。
ヘッド別のαを持っても、各ヘッド内で階層間共有する限り、この条件では同じ性質が残る。
学習済みモデルでは内容logitや層間の文脈が変わるため、この診断だけからモデル全体の
不変性や誤りの原因は結論しない。
[診断コード](experiments/logkv-compressor-20260907/position_probe.py)と
[出力](experiments/logkv-compressor-20260907/position_probe.json)を保存した。

学習可能版Copyingのbest/finalの差に対し、評価精度の補助診断も行った。
T=1〜2,048、各256例・同じseedで、bf16重み＋autocast、fp32重み＋bf16 autocast、
完全fp32（matmul precision=`highest`）を比較した。
T=1の完全一致率はbestがbf16・完全fp32とも93.75%、finalが10.16%・10.55%。
T=64もbestは両方41.02%、finalは13.28%・12.89%だった。
この範囲では、重みのbf16変換を外してもbest/finalの差は解消しない。
長さ全域や他の条件をfp32で再評価した結果ではない。
[診断コード](experiments/logkv-compressor-20260907/precision_probe.py)と
[全出力](experiments/logkv-compressor-20260907/precision_probe.json)を保存した。

## 比較条件

| 条件 | 圧縮係数 | パラメータ数 | Copying GPU | Selective GPU |
|---|---|---:|---:|---:|
| half | 固定 `log(4)/6 = 0.231049…` | 5,786,112 | 0 | 1 |
| base | 固定 `log(4)/3 = 0.462098…` | 5,786,112 | 2 | 3 |
| learned | ヘッド別softplus、初期値は下記 | 5,786,128 | 4 | 5 |

内容logitが等しいとき、最新位置と最古位置の重み比はhalfで2倍、baseで4倍。

learnedの初期値は両Transformer層で共通の等比列で、ヘッド0〜7の順に
`[0.462098, 0.379075, 0.310969, 0.255099, 0.209267, 0.171669, 0.140826, 0.115525]`。
各層・ヘッドの値は独立に学習される。初期化は乱数を消費せず、3条件と前回relative条件の
共通パラメータの初期値が同じであることを確認した。
learnedと固定版では初期のヘッド分布も異なるため、結果の差を「学習可能にした効果」だけに
帰属させる比較ではない。ヘッド別の固定初期値を保つ対照は本実験には含めない。

共通条件はd_model=512、8 heads、d_ff=1024、2層、C=4、KV/V normなし。
各50,000 steps、batch64、AdamW、lr=3e-4、warmup=1,000、weight decay=0、
gradient clipping=1、T~loguniform[1,2028]、全位置のposition-aligned CE。
学習seed0・データseed1、fp32重み＋bf16 autocast。
RTX 3090 24 GiB×6で3条件×2タスクを並列実行した。
GPUごとに速度差があるため、各runの所要時間から方式の速度差は判断しない。

bestは100 stepsごとの訓練区間のstring accuracy、token accuracy、−EMA lossの辞書順で選び、
本評価データは使わない。bestと50,000 stepsのfinalを両方評価した。
評価はseed12345、bf16重み＋autocast、T=1〜131,072の41点、各256例。
入力の長さはT+20で、最後の10桁をtoken正解率と完全一致率で採点する。
前回のphase2とrelativeの完了済み結果も比較基準とする。
すべて同じ評価入力を使い、学習seedは各条件1個。厳密な決定論モードは有効にしていない。

## 成果物と再現方法

`DATA_DIR=/mnt/raid0/RecursiveCompressor`。
実行元は`$DATA_DIR/experiments/logkv-compressor-20260907/source`の固定worktree。
全実行記録とログは同じ親ディレクトリに保存した。
リポジトリ内の[成果物ディレクトリ](experiments/logkv-compressor-20260907/)には、
6 runの設定・学習ログ・best選択記録・評価JSON・保存時の係数、実行コマンドと終了コード、
環境情報、補助診断、集計コードを保存した。図と表は次のコマンドで再生成できる。

```bash
uv run python doc/experiments/logkv-compressor-20260907/summarize.py
```

学習済みモデルの保存先は次のとおり。

```text
$DATA_DIR/exp/<task>/logkv-d512-logu-comp-<mode>-gated-self-20260907/
  model/                 # final
  model_best/            # best
  run_config.json
  train_log.jsonl         # learnedは100 stepsごとの各層・ヘッドの係数も記録
  best.json
  results_best.json
  results_final.json
```

`<task>`は`copying` / `selective-copying`、`<mode>`は`half` / `base` / `learned`。
再実行時は別run nameを使う。base条件のコマンド例:

```bash
export DATA_DIR=/mnt/raid0/RecursiveCompressor
task=copying
run_name=logkv-compressor-reproduction
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run python exp/$task/train.py --run-name "$run_name" --arch logkv \
  --relative-position-bias --compressor-decay 0.46209812037329684 \
  --phase-levels 2 --gated-attention --self-slot \
  --t-dist loguniform --max-t 2028 --steps 50000 --batch-size 64 \
  --lr 0.0003 --warmup 1000 --d-model 512 --num-heads 8 --d-ff 1024 \
  --num-layers 2 --chunk-size 4 --loss-positions all --seed 0 --device 0

for checkpoint in best final; do
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python exp/$task/evaluate.py --run-name "$run_name" --samples 256 \
    --max-t-exp 17 --seed 12345 --precision bf16 --checkpoint "$checkpoint" --device 0
  cp "$DATA_DIR/exp/$task/$run_name/results.json" \
    "$DATA_DIR/exp/$task/$run_name/results_${checkpoint}.json"
done
```

halfでは係数を`0.23104906018664842`にする。
learnedではbaseのコマンドに`--learnable-compressor-decay`を加える。
どの条件も`--phase-emb`は指定しない。
