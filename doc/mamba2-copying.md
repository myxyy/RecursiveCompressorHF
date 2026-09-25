# 公式Mamba-2による固定10桁Copying比較

2026-09-26、LogKVの定数サイズ状態の比較対象としてMamba-2を追加した。
ブランチは`mamba2-copying`。**9月26日01:03:57 JSTに50,000-step学習、best/final計82セルの評価・自動監査が完了した。GPU0は解放済み。**
00:17:35 JST開始、本実行46.4分（最終300-step事前測定から48.5分）。見積もり約2時間、停止期限07:45:28 JST以内に完了した。
[開始記録](experiments/mamba2-copying-20260926/launch.json)、[終了後のCPU再監査](experiments/mamba2-copying-20260926/results/review.json)。
今回はCopyingのみを対象とし、Selective CopyingやLLMの訓練は行わない。

## 完了結果

**Mamba-2は評価したT≤3,072の全30点で各256/256完全一致したが、それより長い距離では低下した。**
bestは36,300 step、finalは50,000 step。訓練区間の指標で選んだbestであり、評価距離別の選び直しはない。
同じ入力例によるLogKV対照は、全41点でbest/finalとも256/256完全一致だった。

| T | Mamba-2 完全一致 best / final（各256例） | Mamba-2 桁精度 best / final | LogKV 完全一致 best / final |
|---:|---:|---:|---:|
| 評価した1～3,072の全30点 | 256 / 256 | 100% / 100% | 256 / 256 |
| 4,096 | 255 / 255 | 99.96% / 99.96% | 256 / 256 |
| 6,144 | 154 / 175 | 95.43% / 96.37% | 256 / 256 |
| 8,192 | 27 / 39 | 82.89% / 84.65% | 256 / 256 |
| 12,288 | 2 / 0 | 63.32% / 62.15% | 256 / 256 |
| 16,384 | 0 / 0 | 42.93% / 50.31% | 256 / 256 |
| 32,768 | 0 / 0 | 14.53% / 34.02% | 256 / 256 |
| 65,536 | 0 / 0 | 12.66% / 17.50% | 256 / 256 |
| 131,072 | 0 / 0 | 13.36% / 12.27% | 256 / 256 |

![Fixed-10 Copying comparison](experiments/mamba2-copying-20260926/results/comparison.png)

最終訓練区間の桁精度・完全一致率はともに100%、EMA lossは2.90e-8だった。
訓練範囲での未収束というより、この設定では訓練範囲の外へ伸ばすと保持性能が崩れる結果になった。
ただし他の初期化・学習率・訓練距離・モデルサイズ・精度なら改善しない、という結論ではない。

### 出力から確認できたこと

T=131,072では**両checkpointとも、それぞれ全256例で回答列が一種類に固定**していた。

- best：`7 8 5 8 5 7 7 1 8 5`
- final：`5 5 3 8 8 5 5 4 2 4`

桁精度13.36% / 12.27%は、数字1..8から一様に推測する12.5%付近。
この評価例では入力ごとの違いがargmax回答に残らなくなっている。
隠れ状態や全logitが完全に同一であることまでは測定していない。

中距離の誤り方もcheckpointで異なる。T=8,192のbestは第4・5桁の誤りが170・145/256例に達する一方、
finalはその2桁が全例正解で、第1・10桁に122・107/256例の誤りが集中する。
固定の桁がアーキテクチャ上必ず識別不能、という証拠にはならない。
また最初の誤りが出るT=4,096は入力全体を一度のstepで処理しており、
8,192トークンの外部分割境界だけで性能低下全体を説明することはできない。

今回の比較では**LogKVが長距離Copyingの完全一致を維持し、Mamba-2は維持できなかった**。
パラメータ数は約579万対345万、学習seed各1個、Mamba-2評価はBF16重み＋autocastなので、
モデル系列全般の優劣や記憶容量の原理的限界を示すものではない。
減衰による情報消失と低精度計算の影響の切り分けには、同じ重みのFP32評価や分割幅比較が次の診断候補となる。
これらの追加実験、16M評価、Selective Copyingは今回実施していない。

### 再監査と成果物

終了後のCPU解析で、全82セルの予測から正答数・桁別誤答を再計算し、marginの符号を照合した。
seedから記憶列を再生成し、best/final間および既存LogKVの保存済み評価例と全点で一致した。
訓練ログ500行とbest選択、重み、凍結ローカルソース、インストール済み上流ソースのハッシュを検証した。
状態保存量は本評価の全点でも1例あたり1,063,936 bytesで一定だった。
今回の完了確認ではGPUを追加使用していない。

- [全82セルの集計・桁別誤答](experiments/mamba2-copying-20260926/results/metrics.json)
- [bestの全予測・margin（gzip JSON）](experiments/mamba2-copying-20260926/results/best.json.gz) / [final](experiments/mamba2-copying-20260926/results/final.json.gz)
- [再監査結果](experiments/mamba2-copying-20260926/results/review.json) / [CPU解析コード](experiments/mamba2-copying-20260926/analyze_completed.py)
- [訓練ログ](experiments/mamba2-copying-20260926/train_log.jsonl) / [完了状態](experiments/mamba2-copying-20260926/campaign.json)

## 追加：LogKVの10億トークン評価との比較

ユーザー提供の`/mnt/raid0/RecursiveCompressor/exp/copying/1b/results.json`を用いて、
今回のMamba-2 best/finalとの比較図を作成した。新たな訓練・GPU評価は行っていない。
**LogKVは記録された全67点、最大T=1,073,741,824（2^30）まで、各8/8例が完全一致**している。
ここでTはCopyingの距離パラメータであり、累計訓練トークン数ではない。固定10桁の入力系列長はT+20。

![LogKV 1b run versus Mamba-2: exact match and digit accuracy](experiments/mamba2-copying-20260926/comparison-1b/comparison.png)

- 左は10桁の完全一致率、右は桁精度。横軸はTの対数表示で、マーカーが実測点を示す。
- LogKVは各8例、Mamba-2は各256例。Mamba-2の線は実測上限T=131,072で終わり、
  その先の灰色領域はMamba-2未評価を示す。0%を10億トークンまで外挿していない。
- 訓練は双方50,000 steps、T上限2,028、固定10桁、2層・d_model512。
  LogKVはCausalConv4・位置埋め込みなし・固定減衰、5,792,256パラメータ。
  Mamba-2は3,445,856パラメータで、同パラメータ数の比較ではない。
- この`1b`は前掲の各256例のLogKV対照とは別の実験。
  提供された集計JSONには評価checkpoint・評価seed・個別予測が含まれないため、
  Mamba-2との評価例の一致や予測からの再集計は確認していない。
  同梱の`best.json`は49,900 stepだが、評価にその重みを使ったとは断定しない。

この記録では、LogKVは訓練距離上限の約53万倍まで10桁を保持できている。
ただし各距離8例の観測であり、任意の入力列での完全一致を保証するものではない。

[PNG](experiments/mamba2-copying-20260926/comparison-1b/comparison.png) /
[SVG（ベクター形式）](experiments/mamba2-copying-20260926/comparison-1b/comparison.svg) /
[描画値CSV](experiments/mamba2-copying-20260926/comparison-1b/comparison.csv) /
[LogKV集計原本のコピー](experiments/mamba2-copying-20260926/comparison-1b/logkv-results.json) /
[出典・SHA256](experiments/mamba2-copying-20260926/comparison-1b/provenance.json)。
67点のLogKV集計と82点のMamba-2集計、入力ファイルのハッシュをCPUで検証して描画した。
再生成は`.venv/bin/python doc/experiments/mamba2-copying-20260926/plot_billion_comparison.py`。

## 実装と公式実装との対応

`models/mamba2/`は公式[`state-spaces/mamba`](https://github.com/state-spaces/mamba)の
`MambaLMHeadModel`を実際に構築し、その`backbone`と`lm_head`を使用する。
依存先はcommit `e9594ce1c732d97440f0332fdc43170a2294dbfa`に固定した（Apache-2.0）。
Mamba-2の係数・射影・畳み込み・gated RMSNorm・初期化を独自の簡易モデルで代替していない。
通常のGPU訓練はそのまま公式backboneのforwardを呼ぶ。

- 純粋なMamba-2ブロック、外側のRMSNorm、FP32残差、最終RMSNorm、入力・出力重み共有。
- `d_intermediate=0`：独立したFFNやattention層は追加しない。
- `d_state=128`、`expand=2`、`headdim=64`、`ngroups=1`、`d_conv=4`。
- A、dtの初期化や出力射影の深さによる再スケールは公式初期化を使う。
- `use_mem_eff_path=False`、`fused_add_norm=False`：公式の非融合経路を使用。
  畳み込みはPyTorch、SSMとRMSNormは公式Tritonカーネル。融合CUDA拡張の速度測定ではない。
- タスクの語彙10に合わせて`pad_vocab_size_multiple=1`を指定する。
  損失は既存Copyingと同じ**位置合わせCE**。通常の次トークンLM損失へのshiftはしない。

`step(input_ids, hidden)`は畳み込み入力の末尾`d_conv-1`個とSSM状態を各層で引き継ぐ。
公式`mamba_chunk_scan_combined`の`initial_states` / `return_final_states`を使い、
複数トークンの継続入力を処理する。1トークンずつのPythonループで本評価を行わない。
入力状態は破壊せず、チャンク境界でdetachもしない。保持状態のサイズは系列長に依存しない。
CPU用の明示的な再帰実装は数値検証用で、本学習には使用しない。
任意のpadding maskやpacked/可変長バッチへの対応はこのラッパーの対象外。

HF形式の`save_pretrained` / `from_pretrained`を用い、保存したモデルをCopying評価器で読み込める。
モデル種別は`mamba2_copying`。数値トークン10語彙のCopying checkpointは日本語生成用のLLMではない。

## 依存環境

```bash
uv sync --extra mamba2
```

通常のLogKV利用にはこのextraは不要。Mamba依存はuv.lockにもcommitを固定した。
上流パッケージがimportするTileLangとの互換性のため`apache-tvm-ffi==0.1.6`も固定している。
0.1.12では本環境のimportが`ir.DictAttrs`の属性登録で失敗した。

Tritonの初回コンパイルにはCコンパイラと使用中Pythonの開発ヘッダーが必要。
本環境ではPython 3.12ヘッダーを、システムへのインストールをせずに
`/mnt/raid0/RecursiveCompressor/python-headers/root/`へ展開し、次を設定した。

```bash
export CPATH=/mnt/raid0/RecursiveCompressor/python-headers/root/usr/include/python3.12:/mnt/raid0/RecursiveCompressor/python-headers/root/usr/include
export TRITON_CACHE_DIR=/mnt/raid0/RecursiveCompressor/cache/triton-mamba2
```

## 比較条件

固定M=10、prefix=0、語彙0..9。訓練Tは1..2,028のloguniform、系列長T+20。
2層・d_model512、3,445,856パラメータ（LogKV対照は5,792,256）、batch64、50,000 steps、AdamW lr3e-4、warmup1,000、weight_decay0、
全位置のCE、勾配clip1、seed0。FP32 master重み＋BF16 autocast。
100-step区間のstring accuracy / token accuracy / EMA lossでbestを選ぶ。
評価距離ごとにcheckpointを選び直さない。

評価はbest/final、T=1..14と2の累乗・中間点の計41点、最大T=131,072、各256例。
両checkpointで同じseed12345のデータを再生し、予測・正解・桁別誤答・marginを保存する。
過去のLogKV評価と同じBF16重み＋autocastを使う。
入力はCPUに生成し、8,192トークンずつGPUで処理する。空白のスキップや状態近似はしない。
SSMカーネル内の`scan_chunk_size=256`は計算タイルであり、受容野の上限ではない。

比較対象は既存[標準LogKV＋CausalConv](logkv-causal-conv.md)の固定10桁評価。
層数・モデル次元・訓練タスク条件を揃えるが、**パラメータ数・状態量・演算精度の詳細・初期化は
同一ではない**。Mamba-2は公式の初期化・重み共有を維持する。
学習seed1個の比較であり、未達の結果もアーキテクチャの限界の証明とはしない。

## 起動方法

```bash
# 単独の訓練
uv run --extra mamba2 python -m exp.copying.train --arch mamba2 \
  --run-name mamba2-copying --d-model 512 --num-layers 2 \
  --max-t 2028 --t-dist loguniform --steps 50000 --batch-size 64 \
  --lr 3e-4 --warmup 1000 --seed 0 --eval-interval 0

# 通常の評価CLIからもcheckpointのモデル種別を判定する
uv run --extra mamba2 python -m exp.copying.evaluate \
  --run-name mamba2-copying --checkpoint best --max-t-exp 17
```

今回の一連の実験は`exp/mamba2_copying/campaign.py`で管理する。
`preflight --root PATH`が300-step速度測定、`run --root PATH`が50k訓練・両checkpoint評価・監査。
起動時にソースをRAID上へ凍結し、8時間以上の見積もりは拒否する。
事前測定から計7.5時間の停止期限を持ち、失敗時の自動再試行や別タスクの追加起動はしない。


## 事前検証・保存先

新規Mamba-2の7テストが通過した。公式との初期重みのbit一致、FP32/BF16のforward比較、
チャンクを跨ぐ勾配、CPU FP64の分割同値性、状態の非破壊・保存量一定、
checkpointの保存・読込、HF generateのキャッシュ付き生成を確認した。
既存回帰テストは332件が初回通過。残る旧モデルのsamplingテストはランダムなEOSで
指定長より短く終了するケースを誤って失敗としていたため、固定8-stepを試すそのテストだけ
EOS停止を無効化し、再実行が通過した（計333件）。モデル本体は変更していない。

初回コンパイル後、最終ソースで300 stepsを再測定した。100→300 stepsの平均は0.0825秒/step。
30%の訓練余裕と30分の評価予備時間を含む全体見積もりは2.00時間。
別の300-step準備checkpointでT=131,072まで全41点×8例の評価器smokeも完了した。
これは未学習に近いcheckpointの実装確認であり、本実験の性能結果ではない。
状態の実保存量は全Tで1例あたり**1,063,936 bytes**。BF16畳み込み履歴とFP32の最終SSM状態を含む。
パラメータ・現在チャンク・作業領域はこの数値に含めない。

ソース・重み・実行ログ・最終評価の保存先：
`/mnt/raid0/RecursiveCompressor/experiments/mamba2-copying-20260926-run/`。
`campaign.json`が完了状態、`train.log`が本学習ログ、
`results/best.json` / `results/final.json`が評価結果。終了時に予測から正答数を再計算し、
両checkpointでの評価例の一致、状態保存量一定、ローカルおよび上流ソースのハッシュを監査した。
原データ・重みはRAIDに保存し、[事前検証記録](experiments/mamba2-copying-20260926/)をリポジトリに残した。


当初の開始確認では1,600 stepsまで正常に進行し、直近100-step訓練区間の完全一致率が100%に到達した。
これは訓練T分布での指標であり、長距離評価の結果ではない。
GPU計算を決定論モードに固定していないため、同一seedの事前測定と本学習の最初300 stepsは
bit一致しなかった。公式との初期重みのbit一致と、学習軌跡のbit再現性は区別する。
[開始後の確認](experiments/mamba2-copying-20260926/start-check.json)。
