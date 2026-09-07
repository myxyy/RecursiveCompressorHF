# 相対位置logitバイアスによる位相埋め込みの置き換え

2026-09-07、実装commit `7261652`。数式・APIは [位置表現実験の設計 §2.4](logkv-position-design.md#24-サブチャンク間の相対位置logitバイアス) を参照。

**全条件完了（2026-09-07）**: 6 runすべての50,000 steps学習とbest/final評価が完了。
retrievalの相対補正だけによる位相埋め込みの置き換えは従来の記憶精度を維持できなかった。
Compressor側の位置バイアスは本実験の対象外。途中記録はcommit `c9d2a05` に残した。

## 提案の検討

コードの最下位をi=0とし、Queryのチャンク内サブユニット番号をj、参照スロットをcとすると、
提案は次のlogit補正で表せる。

```text
logit -= i*log(C) + (j-c-1)*log(C)/(C-1)    # c < j
```

C=4では提示された係数1/3と一致する。提案の「レベル1」はコードのi=0、
「レベル2」はi=1に対応し、i=1の相対項は4トークンごとに同じ値になる。
既存のレベル減衰はそのまま加算し、相対項は全レベルに適用する。
この距離は元トークン単位の距離ではなく、各階層の完成済みサブチャンク間の距離。
直前の完成済みサブチャンクは距離0。self slotは別にバイアス0で扱う。
チャンク境界は絶対位置に固定されるため、系列を1トークンずらしてもattention構造全体が
変わらないという意味での平行移動不変性は持たない。

この固定バイアスは古いスロットの非正規化重みを幾何級数的に減らすため、recency priorとして
妥当であり、因果性・重複なし参照・キャッシュの対数サイズを維持して実装できる。
追加のパラメータやhiddenは不要。圧縮用attention poolingには補正を加えていない。

位相埋め込みとの表現力の同等性は保証しない。例えばすべてのvalueが同じなら、重みだけを
変えても正規化した加重平均は同じであり、絶対位相を新たに作ることはできない。
また、同一レベル内の`j-1`は全候補に共通なので、そのレベル内だけのsoftmaxなら相殺される。
本実装では全レベルとself slotを共通のsoftmaxに含めるため、レベル間の重みには影響する。
圧縮後の順序保持やタスク精度は以下の比較で確認する。

補助診断として、未学習のLogKV（d32、4 heads、C=4、self slotあり、fp64）に同じベクトルを
64回並べて入力した。位置0からの出力の最大差は、noneとrelativeではともに`2.22e-16`、
phase2では`2.26e-2`だった。これは上記の同一valueの性質を数値確認するもので、
記憶タスクの誤りの原因を単独で特定する実験ではない。
[診断コード](experiments/logkv-relative-20260907/position_probe.py)と
[出力](experiments/logkv-relative-20260907/position_probe.json)を保存した。

## 実装と検証

- `relative_position_bias=True` / `--relative-position-bias` をLogKV、Config、LM、学習CLIに追加。
  位相埋め込みを置き換える際は`--phase-emb`を付けない
- 互換性のため新フラグのデフォルトはFalse。既存モデルの設定・重み・hidden形式を変更しない
- C=4の提示行列とlogitバイアスを直接照合。複数レベルをまたぐ正規化の数値も独立に確認
- C=2/3/4、一括・任意分割・逐次推論を独立参照とfp64で比較（誤差1e-12未満）。
  C=2では有効な相対距離が常に0となり、追加補正が出力を変えないことも確認
- 因果性、分割逆伝播、activation checkpointing、学習可能レベル減衰との併用、
  self slot、bf16推論、LMの設定・重みの保存読込を含め、168 tests passed
- GPUでbatch64・系列長2,048のforward/backward/AdamW更新を確認。
  位相埋め込みが存在しないことと有限なloss・勾配を確認した

## 実験条件

すべて現在の重複なし構造で、新規学習した。3条件×2タスクをRTX 3090 24 GiB×6で並列実行。

| 条件 | phase_emb | relative_position_bias | パラメータ数 | Copying GPU | Selective GPU |
|---|---|---|---:|---:|---:|
| relative（提案） | False | True | 5,786,112 | 0 | 1 |
| none（除去のみ） | False | False | 5,786,112 | 2 | 3 |
| phase2（従来） | True、levels=2 | False | 5,794,304 | 4 | 5 |

共通条件はd_model=512、8 heads、d_ff=1024、2層、C=4、gated attention＋self slot、
固定レベル減衰、KV/V normなし。各50,000 steps、batch64、AdamW、lr=3e-4、
warmup=1,000、weight decay=0、gradient clipping=1、T~loguniform[1,2028]、
全位置のposition-aligned CE、学習seed0（データseed1）。fp32重み＋bf16 autocast。

同じタスクの学習データ乱数列は3条件で共通。relativeとnoneは初期化されるパラメータも同一。
phase2は位相埋め込みのパラメータが追加され、その初期化にも乱数を消費するため、
同じseedでも他の重みの初期値はrelative/noneと同じではない。
学習seedは各条件1個で、seed間のばらつきは評価していない。
厳密な決定論モードは有効にしておらず、前回のphase2 runと初期の学習ログには微小な差がある。
今回の対照は新たに学習したphase2とし、前回の評価値は流用しない。

bestと50,000 stepsのfinalを両方評価する。bestは100 stepsごとの訓練区間の
string accuracy、token accuracy、−EMA lossの辞書順で選び、本評価データは使用しない。
評価はseed12345、bf16、T=1〜14の全点＋16〜131,072の2の冪とその1.5倍の計41点、各256例。
最後の10個の数字だけを採点し、token accuracyと10桁すべての完全一致率（string accuracy）を記録する。

## 結果

以下の値はbest / finalの順（%）。

| 指標 | phase2 | none | relative |
|---|---:|---:|---:|
| Copying T=64 完全一致率 | 100 / 100 | 10.16 / 6.64 | 3.52 / 0.78 |
| Copying T=131,072 token正解率 | 100 / 100 | 65.59 / 60.94 | 50.00 / 52.54 |
| Copying T=131,072 完全一致率 | 100 / 100 | 0.39 / 0.39 | 0 / 0 |
| Selective T=64 完全一致率 | 17.58 / 16.41 | 0.39 / 0 | 0 / 0 |
| Selective T=131,072 token正解率 | 25.55 / 26.48 | 34.26 / 32.46 | 18.24 / 18.75 |

phase2のCopyingはbest/finalとも全41点でtoken・完全一致率100%。
Selectiveでは最長Tのtoken精度はnoneがphase2を上回るので、位相埋め込みが全距離で
最良という結果ではない。ただしrelativeはその両者より低い。
今回の比較はretrieval側だけの変更であり、圧縮時にも位置補正を加える案の評価ではない。

| 完了済みrun | bestのstep | finalのstep |
|---|---:|---:|
| phase2 Copying | 49,200 | 50,000 |
| none Copying | 49,800 | 50,000 |
| relative Copying | 49,400 | 50,000 |
| phase2 Selective | 49,000 | 50,000 |
| none Selective | 49,800 | 50,000 |
| relative Selective | 41,000 | 50,000 |

![記憶距離別の比較](experiments/logkv-relative-20260907/comparison.png)

![学習曲線](experiments/logkv-relative-20260907/learning.png)

[代表距離の表](experiments/logkv-relative-20260907/tables.md)、
[全492評価行のCSV](experiments/logkv-relative-20260907/comparison.csv)、
[チェックポイント選択・学習情報](experiments/logkv-relative-20260907/summary.json)を保存した。
100 stepsの訓練区間の精度は異なるTを混ぜた指標であり、保存時点の重みの固定T評価とは異なる。
string accuracyが数割に達していても、それだけで特定の階層の位置認識を分離して評価したことにはならない。

### Copyingの精度・チェックポイント診断

relativeのCopyingはT=1の完全一致率がbestの91.02%からfinalの4.30%へ低下した。
評価時の重みのbf16化だけが原因か確認するため、GPU0で同じ評価入力を使い、
T=1〜2,048の29点を各256例、bf16、fp32重み＋bf16 autocast、fp32で比較した。
fp32はautocastなし、`float32_matmul_precision=highest`。
finalのT=1完全一致率は3方式とも4.30%、T=2,048はすべて0%で、fp32でも問題は解消しなかった。
bf16の全29点は本評価の対応する結果と完全一致した。
[診断コード](experiments/logkv-relative-20260907/precision_probe.py)と
[全結果](experiments/logkv-relative-20260907/precision_probe.json)を保存した。

同一valueの加重平均では位置を区別できないという性質、圧縮時の順序情報、
学習の変動は今後切り分ける必要がある。この1 seedの比較だけで個々の原因は確定しない。

## 成果物と再現方法

実行元は`$DATA_DIR/experiments/logkv-relative-20260907/source`の固定worktree。
`DATA_DIR=/mnt/raid0/RecursiveCompressor`。実行記録と全ログは
`$DATA_DIR/experiments/logkv-relative-20260907/`、学習済みモデルは次に保存される。

```text
$DATA_DIR/exp/<task>/logkv-d512-logu-<mode>-gated-self-20260907/
  model/                 # final
  model_best/            # best
  run_config.json
  train_log.jsonl
  best.json
  results_best.json
  results_final.json
```

`<task>` は`copying` / `selective-copying`、`<mode>` は`relative` / `none` / `phase2`。
再実行時は別run nameを使用する。提案条件のコマンドは次のとおり。

```bash
export DATA_DIR=/mnt/raid0/RecursiveCompressor
task=copying
run_name=logkv-relative-reproduction
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run python exp/$task/train.py --run-name "$run_name" --arch logkv \
  --relative-position-bias --phase-levels 2 --gated-attention --self-slot \
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

noneでは`--relative-position-bias`を外し、phase2ではそれを`--phase-emb`に置き換える。
`--phase-levels 2`はphase2以外では埋め込みを作らず、効果を持たない。
