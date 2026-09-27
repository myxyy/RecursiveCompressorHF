# Mamba-2 + LogKV：固定10桁Copying / Selective Copying

2026-09-28、ユーザー依頼により直列ハイブリッドを実装した。
ブランチは`mamba2-logkv-hybrid`。**9月28日04:29:24 JSTに両タスクの学習・全164セル評価・CPU解析が完了。GPU0/1は解放済み。**
01:14:16 JST開始、並列実行の所要時間は約3時間15分。Copyingの評価完了は04:23:09、Selectiveは04:29:08 JST。
最終事前測定開始からは約3.31時間（その前の実装チェック・初回測定は別）。見積もり約4.7時間・停止期限内に完了した。
[開始記録](experiments/mamba2-logkv-hybrid-20260928/launch.json)。
開始後の確認でCopyingは100 steps、Selective Copyingは200 stepsまで正常に進行し、
凍結ソースのハッシュと事前測定との設定一致を確認した。
[開始確認](experiments/mamba2-logkv-hybrid-20260928/start-check.json)。

## 完了結果

**今回の直列構成では、Copyingの長距離保持を維持しながらSelective Copyingを改善する目標は達成できなかった。**
ハイブリッドは訓練範囲付近では両タスクを解けたが、CopyingはLogKV単体より長距離で大幅に悪化し、
Selectiveも外挿時の完全一致はMamba-2単体を概ね下回った。
以下は同じ評価例・各256例の比較。bestは訓練区間の指標で選択し、距離別の選び直しはない。

### Copying

ハイブリッドのbestは50,000 stepでfinalと同じ重み（safetensorsのSHA256も一致）。
全41点でbest/finalの予測と正答数が一致した。

| T | Hybrid 完全一致 best / final | Mamba-2 完全一致 best / final | LogKV 完全一致 best / final | Hybrid 桁精度 |
|---:|---:|---:|---:|---:|
| 評価した1～3,072の全30点 | 256 / 256 | 256 / 256 | 256 / 256 | 100% |
| 4,096 | 248 / 248 | 255 / 255 | 256 / 256 | 99.65% |
| 6,144 | 96 / 96 | 154 / 175 | 256 / 256 | 90.94% |
| 8,192 | 2 / 2 | 27 / 39 | 256 / 256 | 61.76% |
| 12,288 | 0 / 0 | 2 / 0 | 256 / 256 | 26.33% |
| 16,384 | 0 / 0 | 0 / 0 | 256 / 256 | 14.57% |
| 131,072 | 0 / 0 | 0 / 0 | 256 / 256 | 11.41% |

![Copying comparison](experiments/mamba2-logkv-hybrid-20260928/copying/comparison.png)

最終訓練区間は完全一致100%、EMA loss 2.69e-9で、単なる訓練未収束では説明できない。
T=131,072では256例の回答が2種類に縮退し、223例が`7 2 2 4 2 2 1 7 4 6`だった。
出力の入力依存性が大きく失われたことを示すが、内部状態の完全な消失を測定したわけではない。
今回の重みで16M/1B評価は実施していない。

### Selective Copying

ハイブリッドbestは49,800 step、finalは50,000 step。
bestはT≤2,048の全29評価点で256/256完全一致したが、T=3,072から低下した。

| T | Hybrid 完全一致 best / final | Mamba-2 完全一致 best / final | LogKV 完全一致 best / final | Hybrid 桁精度 best / final |
|---:|---:|---:|---:|---:|
| 1 | 256 / 249 | 256 / 256 | 256 / 256 | 100% / 99.73% |
| 1,024 | 256 / 227 | 256 / 256 | 10 / 19 | 100% / 98.63% |
| 2,048 | 256 / 205 | 256 / 256 | 14 / 30 | 100% / 97.58% |
| 3,072 | 240 / 138 | 256 / 246 | 10 / 29 | 99.14% / 92.77% |
| 4,096 | 191 / 88 | 240 / 216 | 4 / 11 | 94.65% / 86.17% |
| 6,144 | 67 / 11 | 107 / 61 | 11 / 19 | 78.24% / 65.74% |
| 8,192 | 23 / 2 | 26 / 19 | 3 / 11 | 69.02% / 53.71% |
| 16,384 | 1 / 0 | 0 / 0 | 0 / 3 | 41.80% / 32.42% |
| 131,072 | 0 / 0 | 0 / 0 | 0 / 0 | 18.44% / 18.52% |

![Selective Copying comparison](experiments/mamba2-logkv-hybrid-20260928/selective-copying/comparison.png)

**終盤の学習不安定性と長距離外挿の失敗は分けて考える必要がある。**
49,800 stepの100-step区間は完全一致100%、EMA loss 1.19e-6だったが、
49,900で86.98%、50,000で84.61%へ低下し、EMA lossも8.44e-3に上昇した。
finalのT=1での誤答7例は全て第7桁。T=2,048でも第7桁に32/256の誤答が集中した。
bestを選べばこの短距離の悪化は回避できるが、bestでも長距離外挿は改善していない。
原因を過学習と断定できる記録ではなく、終盤の更新による訓練指標・短距離評価の悪化として記録する。

T=131,072の桁精度はMamba-2単体の15.08% / 13.28%より高いが、LogKV単体の41.05% / 48.24%には届かない。
特にbestの第1桁は138/256（53.91%）を保持する一方、残る9桁の合算精度は14.50%。
Mamba-2 bestの第1桁は37/256であり、ハイブリッドの桁精度増加は主に第1桁による。
10桁全体の順序保持が改善したとは解釈できない。

### 考察と次に切り分ける点

LogKVを後段に置いただけでは、長距離メモリを使う解法が学習されるとは限らない。
訓練距離内でMamba-2の状態を利用した解法が成立すれば、その解法に依存したまま外挿に失敗する可能性がある。
今回のCopyingの低下開始距離がMamba-2単体に近いことはこの仮説と整合するが、
LogKVが使われていない、あるいはMamba状態の減衰が原因と直接証明したものではない。

また、現在のLogKVが保存するのはMamba-2で変換・正規化された表現であり、
標準LogKVが学習する表現とは異なる。Keyの区別、Queryとの対応、圧縮後の情報保持が変化した可能性もある。
同時にLogKV側のCausalConvを削除し、出力重み共有・正規化・層数も異なるため、
この比較から一つの変更を原因として選ぶことはできない。
Mamba-2内部の畳み込みはLogKVの各attention直前の畳み込みと同じ役割とは限らない。

次に調べるなら、まず保存済み重みでMamba状態・LogKV階層への介入や読み出しを診断して、
情報が圧縮前後で失われるのか、残っていても取り出せないのかを切り分ける。
新しい構成の候補は、**標準LogKVの入力経路とCausalConvを保持し、別のMamba枝をゲート付きで合流させる形**。
元の経路を維持できる点は利点だが、長距離性能の維持を保証せず、改めてCopyingとSelectiveの両方を評価する必要がある。
これらは追加実験の候補であり、今回の完了確認ではGPU解析・新規学習は起動していない。

### 監査・成果物

CPU解析を再実行し、両タスク50k steps・164セルの予測・margin・best選択・重み・凍結ソース・上流コードを再確認した。
既存LogKVとMamba-2の予測も正答数を再集計し、評価正解列の一致を確認した。
Selectiveの数字配置位置もLogKVの保存済み記録と一致した。
現在のソースと凍結ソース、圧縮保存した予測とRAID原本も一致した。
事前測定と本学習は両タスクともbit一致せず、GPU演算を決定論モードに固定していない点は従来どおり。

- [再監査](experiments/mamba2-logkv-hybrid-20260928/review.json) / [今回の完了確認](experiments/mamba2-logkv-hybrid-20260928/completion-recheck.json)
- [Copying全82セル](experiments/mamba2-logkv-hybrid-20260928/copying/metrics.json) / [Selective全82セル](experiments/mamba2-logkv-hybrid-20260928/selective-copying/metrics.json)
- [桁別誤答・出力縮退の診断](experiments/mamba2-logkv-hybrid-20260928/prediction-diagnostics.json)
- [完了状態](experiments/mamba2-logkv-hybrid-20260928/supervisor.json) / [CPU解析コード](experiments/mamba2-logkv-hybrid-20260928/analyze_completed.py)

各タスク配下の`best.json.gz` / `final.json.gz`に全予測とmargin、`train_log.jsonl`に訓練記録を保存した。

## 構成と仮説

`Embedding → Mamba-2 ×2 → Mamba RMSNorm → LogKVBlock ×2 → RMSNorm → 共有LM head`

Mamba-2で各時点の表現を処理し、その出力を順次LogKVに保存する。
Mambaの最後の状態だけをLogKVに渡す構成ではない。Mamba内部の残差・正規化を維持する。
各LogKVBlockは残差付きattentionとSwiGLU FFNを持つ。

- d_model512、Mamba-2 2層＋LogKV 2層。合計**9,221,728パラメータ**。
- 公式Mamba-2のd_state128 / expand2 / headdim64 / ngroups1 / scan_chunk_size256を維持。
  入出力の埋め込み共有・公式初期化も維持する。LogKV追加部分は既存LogKVBlockの初期化を使用する。
- **Mamba-2内部の幅4の因果畳み込みを使用し、LogKV側のCausalConvは削除**した。
  二つの畳み込みは配置・対象チャンネルが異なり、同じ演算の置換ではない。
- LogKVはC4、8 heads、d_ff1024、位相埋め込みなし、固定レベル減衰、gate/self slotあり。
- 学習は公式Mamba-2の一括forwardを使い、状態を引き継ぐ推論には既存のSSD拡張を使用する。
  hiddenは`{'mamba': 各層の固定サイズ状態, 'logkv': 各層の階層状態}`。
  チャンク境界でdetachしない。保持状態は固定Mamba状態＋対数サイズLogKV状態。

比較対象は既存の[LogKV＋CausalConv](logkv-causal-conv.md)約579万パラメータ、
[Mamba-2 Copying](mamba2-copying.md) / [Selective Copying](mamba2-selective-copying.md)約345万パラメータ。
層数・パラメータ数が増えるため、改善があっても組み合わせ固有の効果とは直ちに断定できない。
最初の実験では組み合わせの実現性と性能を調べ、同パラメータ数の対照訓練は含めない。

## 学習・評価条件

CopyingとSelective Copyingはそれぞれseed0で新規初期化し、独立に50,000 steps学習する。
既存の訓練済み重みは転用しない。固定10桁、T=1..2,028 loguniform、batch64 / accum1、
AdamW lr3e-4 / warmup1,000 / weight_decay0、clip1、全位置の位置合わせCE。
FP32 master重み＋BF16 autocast。タスクごとのデータ生成器・評価seedは従来と同じ。

訓練指標で選んだbestとfinalを、T≤131,072の41点・各256例で評価する（2タスク計164セル）。
BF16重み＋autocast、8,192トークンずつのstep推論。全予測・margin・桁別誤答・状態量を保存する。
Copyingの16M/1B延長はこの実験には含めず、まず131Kまでの性能を確認する。

GPU0でCopying、GPU1でSelective Copyingを並列実行した。使用は最大2GPUで、現在は解放済み。
各タスクで300 stepsと全41点×8例の評価器smokeを事前測定した。
最終測定はCopying 0.2305秒/step、Selective 0.2330秒/step。
それぞれの事前見積もりは4.70 / 4.74時間、終了目安は06:00 JST頃、停止期限は08:10:33 JSTだった。
本番評価の最大バッチ付近（T2048で253例、T8192で63例）も確認し、
評価ピーク割当は11,624,577,536 bytes（約10.83 GiB）、出力は有限だった。
学習30%の余裕と、smokeから推定した評価時間（最低30分）を含めて見積もる。
見積もり8時間以上なら開始せず確認する。事前測定開始から7時間で強制停止し、
それ以前の実装GPUチェック分も含め8時間以内に収める。自動再試行・追加seedは起動しない。

## 検証と使い方

既存Mamba-2と新規ハイブリッドを合わせて13テストが通過。
公式Mamba-2と初期重みが一致し、LogKV側に畳み込みがないことを検証した。
CPU FP64で任意分割の出力・状態・勾配が一致し、入力状態が非破壊であること、
保持テンソルが入力全体のストレージを保持しないことを確認した。
GPU FP32 / autocast / BF16重みで公式学習経路と分割推論を照合し、FP32勾配も照合した。
HF保存・再読込、モデルディレクトリ指定の読込、キャッシュ付きgenerateも通過した。

本番サイズのbatch64・T2028で2回の更新を確認し、全パラメータの勾配が有限だった。
初回の可変T事前測定ではSelective Copyingにメモリ断片化によるOOMが発生した。
初回成果物は別フォルダに保存し、両タスクで`PYTORCH_ALLOC_CONF=expandable_segments:True`を設定して
再測定し、両タスクとも学習・長距離評価の事前確認を通過した。
モデル・バッチサイズ・データ条件は変更していない。
ピーク割当は20,677,343,232 bytes（約19.26 GiB）。

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
uv run --extra mamba2 python -m exp.copying.train --arch mamba2-logkv \
  --run-name hybrid-copying --d-model 512 --mamba-num-layers 2 --num-layers 2 \
  --num-heads 8 --d-ff 1024 --conv-kernel-size 0 \
  --steps 50000 --max-t 2028 --t-dist loguniform --batch-size 64 \
  --lr 3e-4 --warmup 1000 --seed 0 --eval-interval 0

# Selective Copyingは入口をexp.selective_copying.trainに変更する。
# --num-layersはLogKV層数、--mamba-num-layersは前段のMamba-2層数。
uv run --extra mamba2 python -m exp.copying.evaluate --run-name hybrid-copying \
  --checkpoint best --max-t-exp 17 --samples 256
```

`models/mamba2_logkv/`にモデルと設定、`exp/mamba2_logkv/`に予測保存付き評価器とキャンペーンを追加した。
`--arch mamba2-logkv`ではLogKV側conv幅の既定が0。他の既存アーキテクチャの既定値は維持する。
モデル種別`mamba2_logkv`を共通評価器・`inference.predict`のローダーから認識できる。
今回のcheckpointは数字10語彙のタスク用であり、日本語生成用LLMではない。

大型成果物・凍結ソースは`/mnt/raid0/RecursiveCompressor/experiments/mamba2-logkv-hybrid-20260928-run/`。
各タスク配下の`campaign.json`が状態、`train.log`が学習、`results/{best,final}.json`が評価結果。
両タスク完了後、CPUで既存LogKV/Mamba-2との評価例の一致・予測・重み・ソース・状態量を監査し、
比較グラフを[実験記録](experiments/mamba2-logkv-hybrid-20260928/)に保存した。
