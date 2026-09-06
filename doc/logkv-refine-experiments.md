# 重複除去後の Copying / Selective Copying 評価

2026-09-06〜07、旧構造 `c9c51c9` と重複除去後の `3b0ce51` を同条件で新規学習した。
変更内容は [logkv.md §6.17](logkv.md#617-重複なしの階層参照2026-09-06) を参照。

新構造はCopyingの完全コピーを維持し、固定サイズ測定でピークメモリが約13%減った。
Selective Copyingは旧構造より精度が低下した。学習seedは各条件1個。

![長さごとの評価](experiments/logkv-refine-20260906/comparison.png)

Copyingの4本の曲線はすべて100%で重なっている。全数値は
[CSV](experiments/logkv-refine-20260906/comparison.csv)を参照。

## 実験条件

| 項目 | 設定 |
|---|---|
| モデル | LogKV、d_model=512、8 heads、d_ff=1024、2 layers、C=4 |
| オプション | phase_emb / phase_levels=2、gated attention、self slot、固定レベル減衰、KV/V normなし |
| パラメータ数 | 各5,794,304 |
| 学習 | 各50,000 steps、batch=64、gradient accumulation=1 |
| 最適化 | AdamW、lr=3e-4、warmup=1,000、weight decay=0、gradient clipping=1 |
| データ | Tをloguniformで1〜2,028から抽出、系列長T+20、全位置のposition-aligned CE |
| 精度 | 学習はfp32パラメータ＋bf16 autocast、評価はbf16 |
| seed | 学習0（データ生成1）、本評価12,345 |
| 本評価 | T=1〜14の全点と、16〜131,072の2の冪・その1.5倍、計41点、各256例 |
| checkpoint | bestとfinalの両方。bestは100 stepsごとの訓練区間のstring accuracy、token accuracy、−EMA lossの辞書順で選択 |
| ハードウェア | RTX 3090 24 GiB × 4で4条件を並列実行。追加評価・測定には空きGPUを使用 |

Copyingは冒頭の10個の数字（1〜8）を空白区間後に順に出力する。
Selective Copyingは最初のT+9位置に散在する10個の数字を出現順に出力する。
評価はどちらも最後の10位置に限定し、token accuracyは数字ごとの正解率、
string accuracyは10個すべてが正しい割合を示す。空白位置の正解率は含めない。
入力中のマーカーに対する出力を採点し、正解の数字は入力にフィードバックしない。

新旧で同じタスクの設定はrun name以外すべて一致させた。学習データと評価データの乱数列も
新旧間で共通にし、各構造を固定した別worktreeから実行した。
bestの選択には本評価データを使用していない。学習seedは各条件1個のため、
以下はこの条件・seedにおける結果であり、seed間のばらつきは未測定。

## 学習曲線

| Task | 構造 | GPU | best step | final EMA loss | 学習時間（分） |
|---|---|---:|---:|---:|---:|
| copying | 旧 | 2 | 48,100 | 4.93336e-06 | 178.7 |
| copying | 新 | 0 | 49,700 | 6.65272e-10 | 137.6 |
| selective-copying | 旧 | 3 | 49,000 | 0.0116403 | 153.2 |
| selective-copying | 新 | 1 | 49,000 | 0.0232249 | 141.1 |

![学習曲線](experiments/logkv-refine-20260906/learning.png)

Copyingで100 steps区間の完全一致率が初めて99%以上になったのは、旧5,300 steps、新10,900 steps。
これはその後の安定性を保証する指標ではない。特に旧Copyingでは後半にも損失の急増があり、
48,000 steps付近の低下から最終区間では完全一致率100%に回復した。
学習中の不安定さを含めて確認できるよう、bestだけでなくfinalの評価も残した。

## Copying

新旧ともbest / finalの全4モデルが、本評価の全41点でtoken・string accuracy 100%。
T=131,072でも各256例すべてを正しくコピーできた。

追加プローブでは、新構造のfinalをT=2^18〜2^24の7点、各8例、seed54321で評価し、
全点でtoken・string accuracyとも100%だった。最大T=16,777,216で8/8。
推論は本評価と同じ8,192トークンごとのstep呼び出しでhiddenを引き継ぐ。
この小標本の追加評価は新構造のみで実施し、旧構造との性能差や、未評価の長さでの成功率は示さない。

## Selective Copying

値は **token / string accuracy（%）**。新旧ともbestは49,000 steps、finalは50,000 steps。

| T | 旧 best | 新 best | 旧 final | 新 final |
|---:|---:|---:|---:|---:|
| 16 | 98.40 / 85.94 | 96.72 / 72.27 | 98.83 / 89.45 | 94.41 / 56.64 |
| 64 | 88.12 / 30.47 | 84.38 / 17.58 | 88.09 / 31.25 | 80.23 / 12.89 |
| 256 | 80.51 / 9.38 | 75.74 / 3.91 | 79.45 / 7.42 | 72.70 / 3.12 |
| 1,024 | 74.61 / 2.73 | 68.63 / 1.56 | 74.26 / 3.52 | 61.72 / 0.39 |
| 2,048 | 68.55 / 2.73 | 65.90 / 0.78 | 70.98 / 0.39 | 59.57 / 0.39 |
| 8,192 | 59.84 / 0.78 | 56.76 / 0.39 | 63.32 / 0.39 | 50.31 / 0.00 |
| 32,768 | 50.47 / 0.00 | 43.87 / 0.00 | 53.52 / 0.00 | 31.76 / 0.00 |
| 131,072 | 40.66 / 0.00 | 32.77 / 0.00 | 44.88 / 0.00 | 20.43 / 0.00 |

T=64のbest完全一致率は30.47%→17.58%（−12.89ポイント）、
T=131,072のbestトークン正解率は40.66%→32.77%（−7.89ポイント）。
finalでも同じ方向の差があり、この条件では重複除去によるSelective Copyingの改善は見られない。
bestは訓練区間から選んだもので、各Tにおいてfinalより高い精度を保証するものではない。

## 固定サイズでの計算コスト

| 構造 | forward＋backward平均 | ピークallocated VRAM |
|---|---:|---:|
| 旧 | 1.198秒 | 16.964 GiB |
| 新 | 1.138秒 | 14.763 GiB |

同一GPU（物理GPU 4）で旧→新の順に測定。batch=64、系列長=2,048、上記と同じモデル構成、
ランダムな入力、fp32パラメータ＋bf16 autocast。3回warmup後、CUDA同期を挟んで10回計測した。
forward、loss.backward、zero_gradを含み、optimizer stepは含まない。
所要時間は約5.0%、ピークallocatedメモリは約13.0%減少した。
この単一サイズでの測定値であり、GPUの異なる本学習の所要時間から速度比を推定しない。

Python 3.12.3、PyTorch 2.11.0+cu128、Transformers 5.4.0、CUDA 12.8、
NVIDIA driver 595.84、OMP/MKL threads=1、float32_matmul_precision=high。

## 解釈と残る検証

Copyingでは新構造でも非常に長い空白区間を越えて記憶を保持できた。
一方、今回のSelective Copyingでは訓練域内・域外とも精度低下が見られるため、
重複除去を品質改善と結論づける結果ではない。

下位の表現を参照できる期間が短くなった影響や、固定レベル減衰が新構造にも適切かは
今回の比較だけでは切り分けられない。次は複数seedでの再現性確認と、
同じ新構造での固定レベル減衰の有無の比較が候補になる。
LMの再訓練・生成品質と、位相やnorm等の再調整は今回評価していない。

## 再現方法と成果物

学習済み重みと全ログは次に保存した。`<task>` は `copying` または `selective-copying`、
`<layout>` は `refined` または `overlap`。

```text
$DATA_DIR/exp/<task>/logkv-d512-logu-ph2-gated-self-<layout>-20260906/
  model/                 # final
  model_best/            # best
  best.json
  run_config.json
  train_log.jsonl
  results_best.json
  results_final.json
  plot_best.png
  plot_final.png
```

本実行の `DATA_DIR` は `/mnt/raid0/RecursiveCompressor`。
実行コマンド、終了コード、ソースのcommit、環境情報、集計スクリプトは
`$DATA_DIR/experiments/logkv-refine-20260906/` に保存した。

リポジトリ内の [experiments/logkv-refine-20260906/](experiments/logkv-refine-20260906/) にも、
4条件の実行記録、設定、学習ログ、best/finalの評価JSON、追加プローブ、計算コストの生データを
保存した。学習済み重みを読み込まず、保存済みデータだけで図・CSVを再生成できる。

```bash
uv run python doc/experiments/logkv-refine-20260906/summarize.py
```

各構造の対応commitで、空いているGPUを指定して以下を実行する。
旧構造の重みは必ず旧commitで評価する（現行コードに読み込むとattentionの意味論が変わる）。

```bash
# task、layout、GPU番号を各条件に合わせて設定する
export DATA_DIR=/mnt/raid0/RecursiveCompressor
task=copying
layout=refined
gpu=0
run_name=logkv-d512-logu-ph2-gated-self-${layout}-20260906

CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run python exp/$task/train.py --run-name "$run_name" --arch logkv \
  --phase-emb --phase-levels 2 --gated-attention --self-slot \
  --t-dist loguniform --max-t 2028 --steps 50000 --batch-size 64 \
  --lr 0.0003 --warmup 1000 --d-model 512 --num-heads 8 --d-ff 1024 \
  --num-layers 2 --chunk-size 4 --loss-positions all --seed 0 --device 0

for checkpoint in best final; do
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python exp/$task/evaluate.py --run-name "$run_name" --samples 256 \
    --max-t-exp 17 --seed 12345 --precision bf16 --checkpoint "$checkpoint" --device 0
  cp "$DATA_DIR/exp/$task/$run_name/results.json" \
    "$DATA_DIR/exp/$task/$run_name/results_${checkpoint}.json"
  cp "$DATA_DIR/exp/$task/$run_name/plot.png" \
    "$DATA_DIR/exp/$task/$run_name/plot_${checkpoint}.png"
done
```

既存run directoryへの上書きを避けるため、再実行時は別のrun nameを使う。
