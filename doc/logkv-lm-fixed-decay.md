# 標準LogKV言語モデル：固定減衰と学習可能減衰の再学習比較

2026-09-16、ユーザーの依頼により、[学習可能減衰の5k LM実験](logkv-lm-learnable-decay.md)に
対応する固定減衰モデルを新規学習する。**2026-09-16 10:25:40 JSTに6GPUで本学習を開始した。結果は未確定。**
準備commit `e461e5e`。事前見積もり5.45時間、終了目安は16時頃、停止上限17:50:59 JST。
[実行・監視記録](experiments/logkv-lm-fixed-decay-20260916/campaign.json)。
作業ブランチは`adjust-attenuation`。ユーザーは今回も全6GPUの使用を許可した。
8時間以上の見込みなら開始前に確認する。今回の実測見積もりは約5.5時間。

## 比較条件

| 項目 | 固定減衰（今回） | 学習可能減衰（完了済み対照） |
|---|---|---|
| 凍結ソース | 前回と同一の`ce360e3`の7ファイル | 同左 |
| モデル | d1024/H8/ff3072/16層/C4、context2048 | 同左 |
| 構造 | 重複なし、phaseなし、CausalConv幅4、gate/self slotあり | 同左 |
| logit補正 | `−i log(4)` | `−i β[layer,head]`、β初期値log(4) |
| パラメータ数 | 327,392,256 | 327,392,384（+128） |
| 学習 | 5,000 optimizer steps、新規seed0、6GPU×batch4×蓄積1 | 同左 |
| データ | 同じpretrain mmap、同じDistributedSampler順序、120,000 rows | 同左 |
| 最適化 | Muon＋AdamW、lr2e-4、warmup1000、clip1 | 同左 |
| 精度 | fp32 master＋bf16 autocast、評価も同じ | 同左 |
| 保存・生成 | 1000 stepsごと、5個のcheckpoint保持、固定日本語3prompt | 同左 |
| 周期サンプルRNG | `12345+step`、訓練RNGに影響させない | 同左 |

`--learnable-decay`を指定せず、元の標準固定減衰を用いる。その他のCLI設定差はrun名のみ。
前回と同じ未学習初期化をCPUで再構成し、学習可能版／固定版の**共通する全tensorが完全一致**する
ことを確認した。実際の固定版初期重みも同じfingerprintに一致することを各run開始時に検証する。
学習済み重みの転移や前回runの再開ではない。
[初期化照合](experiments/logkv-lm-fixed-decay-20260916/initialization_audit.json)。

**比較の範囲**：元の固定バイアスはPython scalar、学習可能バイアスはfp32 tensorで、
bf16 autocast下のlogit加算の精度も異なる。既存実装を変えずに比較するため、結果はこの数値計算差を
含むオプション全体の差。係数学習だけの効果を厳密に分離した実験ではない。
学習可能版の推論時βリセットよりは、固定減衰に適応する再学習を含めた対照になる。

## 同じ入力・条件での評価

- 前回と同じ未消費packed row 128例（4ソース各32例）、212,943教師tokenのloss/PPL。
  入力・教師のハッシュ、行index、PADマスクが一致することを確認する。
  行単位で訓練と分離しているが、文書単位の重複除去をした外部評価セットではない。
- 日本語3prompt×seed0–2×temperature0.7/1.0、最大1024新規tokenの18件。
  同じ3promptのseed0/temp0.7を最大4096新規tokenまで延長する3件。合計21件。
  top_p=0.9、repetition_penalty=1、EOSで停止。生成tokenと全文を保存する。
- EOS、長さ、末尾1/4の文字bigram異なり率、token4gram反復率、同一token連続数を比較する。
  同じseedでもモデルが違えば生成は分岐する。短いEOSや指標だけで生成品質を判断しない。
- attentionは前回の修正後と同じ128例・最後の最大512**有効query位置**を使う。
  入力が非PADかつ教師が非maskedの位置のみ、計64,338 query/ヘッド。
  全層・ヘッドのレベル別／自己スロットへの質量を比較する。gate・出力射影前の値で、
  最終出力への因果的寄与を直接表すものではない。
- 全5000-step loss/grad/lr、5個の保存checkpoint、初期化、生成指標と原文を監査する。
  訓練loss/EMAは両方ともrank0の値で、6GPU平均ではない。単一seedの比較。

前回のPADを含んだ旧`results/attention_mass.json`は比較に使用しない。
`analysis/attention_valid.json`の修正済み対照を使い、今回も観測器の有無で出力がbit一致することを確認する。

## 実行管理

前回と同条件の実データ30-step benchmarkと、保存済み30-stepモデルでの評価器・attention観測器確認を
本学習前に実施する。[実測記録](experiments/logkv-lm-fixed-decay-20260916/preflight.json)。
予算は実測訓練時間に15%の余裕＋評価1時間。事前GPU実測開始から7.5時間を全体停止上限とする。
6GPUの訓練後はGPU0のみで評価し、CPUで対照との集計・監査を実施する。失敗時は停止し、再試行や
追加訓練はキューしない。終了後に全GPUを解放する。

実験コード・開始/終了記録は[実験フォルダ](experiments/logkv-lm-fixed-decay-20260916/README.md)。
大きな成果物は`/mnt/raid0/RecursiveCompressor/experiments/logkv-lm-fixed-decay-20260916/`。
checkpointはその下の`data/checkpoints_logkv/d1024-h8-l16-conv4-fixed-decay-5000/`。
前回の学習可能版とユーザーの別runは保存したまま、mainのモデル・CLIは変更しない。
