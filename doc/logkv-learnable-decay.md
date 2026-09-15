# CausalConv＋ヘッド別学習可能レベル減衰：固定10桁比較

2026-09-15、ユーザーの依頼により既存の`--learnable-decay`を用いた
Copying / Selective Copyingの独立した再学習・評価を実施する。
モデル本体・通常CLI・標準構成は変更せず、main `0e2e949`のコードを凍結する。
**状態：事前検証・GPU間再現性・評価器と係数ログの照合を通過。本学習開始前、結果は未確定。**

見積もりは事前GPU確認込み約4.30時間（学習速度に15%の余裕＋評価予算1時間）。
最大使用メモリ15.85 GiB、300 stepsはCopying61.4秒・Selective57.8秒。
事前GPU確認は2026-09-15 16:37:56 JST開始、停止期限は9月16日00:07:56 JST。
[benchmark](experiments/logkv-learnable-decay-20260915/preflight.json)、
[評価器確認](experiments/logkv-learnable-decay-20260915/evaluation_smoke.json)、
[係数更新と保存値の照合](experiments/logkv-learnable-decay-20260915/coefficient_smoke.json)。

今回の比較は固定10桁・prefixなし。直近の可変桁実験とは分け、
[固定減衰のCausalConv実験](logkv-causal-conv.md)を対照とする。
対照の学習済み重みは転移せず、保存済みの同じ**未学習**重みから新規学習する。
新たに比較する条件は学習可能係数の1種類のみ。対照の再学習、補正なし、固定増幅、
ゼロ初期化、可変桁、LM訓練、16M延長は今回の範囲に含めない。

## 係数と比較条件

各attention層・ヘッドに1個の係数βを持ち、レベルiのlogitに`−i β`を加える。
2層×8ヘッド＝16パラメータで、圧縮レベル間では共有する。

- 初期値：β=log(4)。現行の固定減衰から開始し、ゼロ初期化はしない。
- β>0：減衰、β=0：補正なし、β<0：増幅。符号制約・clamp・追加正則化はない。
- 学習率・AdamW・weight decay0・勾配clipはほかのパラメータと共通。
- 100 stepsごとに2×8のβとα=−β/log(4)を記録する。best/finalの保存重みとも照合する。
- 自己スロットは従来どおりレベル補正0。

| 項目 | 固定減衰の対照 | 今回 |
|---|---|---|
| 構造 | 重複なしLogKV、Conv幅4、位相埋め込みなし | 同左 |
| モデル | d512/H8/ff1024/2層/C4、gate/self slotあり | 同左 |
| レベルlogit補正 | −i log C | −i β、層・ヘッド別に学習 |
| パラメータ数 | 5,792,256 | 5,792,272（+16） |
| 記憶長・prefix | M10固定・P0 | 同左 |
| 学習 | 各50k steps、batch64、蓄積1、T loguniform[1,2028] | 同左 |
| 最適化 | AdamW lr3e-4、warmup1000、weight decay0、clip1 | 同左 |
| seed | 初期化0・訓練データ1・最終評価12345 | 同左 |
| 損失・best選択 | 全位置CE、100-step訓練区間のstring/token/−EMAの辞書順 | 同左 |
| 精度 | 学習fp32重み＋bf16 autocast、評価bf16重み＋autocast | 同左 |
| 最終評価 | best/final、T131072まで41点×各256例 | 同左、計164セル |

共通するすべての重み（Convを含む）を対照の初期checkpointからコピーし、βだけを追加する。
2タスクで同じ初期状態を使用するが、学習・checkpoint・評価は独立。
実行設定の差はrun名・learnable_decay・パラメータ数のみであることを監査する。

**既存オプションの数値計算上の差**：固定バイアスはPython scalar、学習可能バイアスは
パラメータtensorであり、fp32重み＋bf16 autocast時には後者のlogit加算がfp32へ昇格する。
そのため、共通重みと理論上の初期係数を揃えても初期出力のbit一致は主張しない。
これは既存`--learnable-decay`の挙動であり、本実験では変更しない。性能差は
この精度差を含むオプション全体の差で、係数を学習する効果だけの厳密な分離ではない。
評価時はモデル全体をbf16にするため、βもbf16へ丸められる。fp32保存値と評価値を併記する。

## 検証・実行管理

CPUでβを正確なlog Cに設定したfp64固定モデルとの一致、負・ゼロ・正のβを混ぜた
forward/step/predict一致、8係数の有限差分による勾配検証を通過した。
[検証記録](experiments/logkv-learnable-decay-20260915/validation.json)。
GPUで最大batch64/T2028の更新、混合精度と逐次推論、20-stepのGPU間再現性、
各300-stepのbenchmarkを行い、所要時間を見積もる。

GPU0=Copying、GPU1=Selective、最大2台。見積もりが8時間以上なら本学習前に確認する。
停止上限は事前GPU検証開始から7.5時間。失敗時は両workerを停止し、自動再試行しない。
全評価の正答数・入力記憶・実配置を元タスク生成器で独立再生し、対照と同じ例での
改善・悪化を保存する。βの推移・best/finalの符号と大きさも監査する。
この一連の実験が終わったら停止し、追加条件はキューしない。

保存先：`/mnt/raid0/RecursiveCompressor/experiments/logkv-learnable-decay-20260915/`。
[実行コード](experiments/logkv-learnable-decay-20260915/README.md)、
[凍結ソース記録](experiments/logkv-learnable-decay-20260915/source_manifest.json)。
