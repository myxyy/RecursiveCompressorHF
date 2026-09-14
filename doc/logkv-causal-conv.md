# LogKV：位置埋め込みなし＋CausalConvの比較

2026-09-14、ユーザーの依頼によりmain `d707675`から`logkv-causal-conv`ブランチを作成した。
**2026-09-14 11:04:31 JSTにGPU0/1で本学習を開始した。結果は未確定。**
両タスクの最初の100 stepsは事前benchmarkと記録指標がbit一致した。
対照とのrun設定差分はconv幅・パラメータ数・run名だけであることも確認した。
[開始確認](experiments/logkv-causal-conv-20260914/launch.json)。準備commit `dab79f2`。
評価込み約4.3時間、完了は**本日15:20頃**が目安。停止期限は18:30:42 JST。
mainへのモデル変更のマージは行っていない。
実装commit `4bfee1f`の28ファイルを実験専用ディレクトリへ凍結した。
モデル・通常訓練CLI等の変更は同commitで追跡し、タスク生成・評価器は既存対照と同一。
[ソース記録](experiments/logkv-causal-conv-20260914/source_manifest.json)。

## 実装

各LogKVBlockで、従来のattention直前に次の残差を追加する。

```text
x ← x + SiLU(DepthwiseCausalConv(RMSNorm(x)))
x ← x + LogKV(RMSNorm(x))
x ← x + FFNSwiGLU(RMSNorm(x))
```

- 今回は幅4・stride1・dilation1、各チャネル独立、biasあり。各Blockは別のconvを持つ。
- 適用するのはBlock入力のトークン列。Q/K/Vはconv残差を加えた後の表現から作る。
- 再帰圧縮の各レベルへ追加適用する方式ではない。圧縮処理・参照スロット規則・固定−i log Cは維持する。
- convが近隣の情報を混合するため、各スロットの内容の受容野は広がる。元トークン内容まで完全に非重複という主張ではない。
- 全体の先頭だけをゼロpaddingし、チャンク境界・`step`呼び出し境界では履歴を引き継ぐ。
- 追加hiddenは各Blockにつき直近3個の正規化済み入力。小さい履歴をcloneし、巨大な元テンソルのstorageを保持しない。
- 追加キャッシュは系列長によらずO(k d)、LogKV全体のO(log L)キャッシュを維持する。
- `conv_kernel_size=0`は既定・無効。旧checkpointのパラメータ構成とhidden形式を維持する。
  有効時のBlock hiddenは`(conv_cache, attention_hidden)`。実行中に設定を切り替えず、別設定ではhidden=Noneから開始する。
- Copying/Selectiveと通常LM訓練のCLIは`--conv-kernel-size 4`。LM用Muonへ3次元のconv重みを渡さずAdamWへ割り当てる。

300 stepsの実checkpointで両タスクbest/finalをT3/16/65・各256例で評価し、
記憶列・実配置・正答数の独立照合を通過した。対照との同一例比較の集計も検証した。
[評価器確認](experiments/logkv-causal-conv-20260914/evaluation_smoke.json)。

## 比較条件

対照は完了済みの[2層・位置埋め込みなし](logkv-no-position-main.md)を用いる。
3層化・損失関数変更・新しい位置表現は併用せず、conv追加による差を調べる。

| 項目 | 対照 | 今回 |
|---|---|---|
| 構造 | 重複なしLogKV、2層 | 同左＋各Blockのconv残差 |
| C / d / heads / d_ff | 4 / 512 / 8 / 1024 | 同左 |
| 位相埋め込み | 無効 | 無効 |
| conv幅 | 無効 | 4 |
| gate / self slot / 固定−i log C | 有効 | 同左 |
| KV/V norm・level amplification | 無効 | 無効 |
| パラメータ数 | 5,786,112 | 5,792,256（+6,144、約0.11%） |
| タスク | 固定10桁Copying / Selective | 同左、別モデルとして学習 |
| 学習 | 各50k steps、batch64、grad accumulation1 | 同左 |
| T分布・損失 | loguniform[1,2028]・全位置CE | 同左 |
| 最適化 | AdamW、lr3e−4、warmup1000、weight decay0、clip1 | 同左 |
| seed | 初期化0、訓練データ1、評価12345 | 同左 |
| 精度 | 学習fp32重み＋bf16 autocast、評価bf16重み＋autocast | 同左 |
| 評価 | best/final、T131072まで41点、各256例 | 同左、新規164セル |

conv付きモデルをseed0で生成後、共通部分は対照の保存済み**未学習**重みと全テンソル一致させる。
追加convのnorm・重み・biasだけを新規初期化から残し、この初期状態を2タスクで共用する。
学習済みcheckpointの転移ではない。bestは訓練100 steps区間のstring/token/−EMA lossの辞書順で選ぶ。
同じタスク内で2条件のデータseed・batch・乱数消費手順・評価例を揃える。

## 検証と実行管理

CPUテスト159件を通過。独立した過去tap演算＋既存の独立attention参照、fp64のforward/step/predict一致、
未来入力への非依存、分割逆伝播の一致、定数サイズconv履歴とstorage、HF保存読込・生成、
bf16重み推論、conv無効の旧config読込、Muon/AdamW振分けを確認した。
保存読込は全重みの完全一致とfp64出力誤差1e−12以内を検査する。
[テスト記録](experiments/logkv-causal-conv-20260914/tests.json)。

GPU事前検証は2026-09-14 11:00:42 JST開始。最大T2028・batch64で2回のoptimizer更新を通過し、
ピークallocated 15.85 GiB、reserved 16.49 GiB。GPUのfp64分割推論最大差は5.6e−17、
確認した小型モデルでfp32＋bf16 autocast・bf16重みの分割推論は最大差0だった。
conv無効モデルの初期重みとfp64出力も、凍結した2層対照とbit一致した。
[最大サイズ・GPU逐次推論確認](experiments/logkv-causal-conv-20260914/fullsize_smoke.json)、
[ソース・初期化・互換性確認](experiments/logkv-causal-conv-20260914/validation.json)。

20 stepsのGPU0/1反復は両タスクで重み・記録指標がbit一致。
300 stepsの同時実測はCopying 61.7秒、Selective 57.7秒で、50kへの単純外挿は2.86 / 2.67時間。
15%の学習時間余裕・評価1時間・GPU事前検証を含め、全体見積もりは**4.31時間**。
事前GPU検証を含む停止期限は**2026-09-14 18:30:42 JST**。
[再現性・時間計測](experiments/logkv-causal-conv-20260914/preflight.json)。

GPU0=Copying、GPU1=Selectiveで最大2GPU。まずGPUでの分割推論・最大T/batchの更新を確認し、
20-step反復と300-step同時benchmarkで再現性・所要時間を確認する。
8時間以上の見積もりなら実行前に確認する。本batchはGPU事前検証開始から7.5時間の上限で管理する。
失敗時は今回のworkerと子プロセスを停止し、追加実験はキューしない。

完了後は保存予測から正答数・桁別誤答を再計算し、元タスクgeneratorで記憶列・配置を再生して照合する。
対照と同じ例で、完全一致へ改善した数と逆方向へ悪化した数、桁別の改善/悪化を保存する。
特にT16の第7桁、T64の第8・9桁、長距離の桁の反復・取り違えに注目する。

1 seedの比較で、長距離完全一致や旧アーキテクチャのconv効果の再現を保証するものではない。
長い同一入力区間でconv窓が同じになればconv出力も同じになる制約は残る。
今回の結果で局所特徴追加の効果を調べ、圧縮階層ごとのconv等は別の実験として扱う。

大型生成物：`/mnt/raid0/RecursiveCompressor/experiments/logkv-causal-conv-20260914/`。
[実行コード](experiments/logkv-causal-conv-20260914/README.md)。
