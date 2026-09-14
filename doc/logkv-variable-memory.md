# 標準CausalConv：可変桁Copying / Selective Copying

2026-09-14、固定10桁Copyingの16M評価成功を受け、ユーザー承認により
`logkv-causal-conv`をmainへfast-forwardマージした（標準化commit `654f311`）。
新規訓練CLIはConv幅4・位相埋め込みなし・gate/self slotありを既定とする。
旧checkpointを読む低水準configの既定値は互換性のため維持する。
モデルテスト159件、CLIの標準設定・無効化オプションの確認を通過した。

続くユーザー依頼は、[以前の可変桁実験](logkv-position-study.md)と同条件で
新しい標準構成を学習・評価すること。今回新しく学習する構成はCausalConvのみで、
CopyingとSelectiveを別モデル・別GPUとして並列に実施する。

**状態：事前benchmark・440セルのCPU再生照合を通過。本学習を開始する準備が完了した。最終結果は未確定。**

見積もりは事前確認込み約5.08時間。最大形状の使用メモリ8.98 GiB、300 stepsはCopying64.18秒・Selective64.25秒。
220セル×16例の評価は109.58秒・108.27秒。学習50k、25回の検証、best/final各256例へ換算し、
25%の余裕と5分の予備時間を加えた。停止期限は2026-09-14 22:02:35 JST。
[benchmark記録](experiments/logkv-variable-memory-20260914/preflight.json)、
[評価データのCPU照合](experiments/logkv-variable-memory-20260914/evaluation_smoke.json)。

## 条件

| 項目 | 設定 |
|---|---|
| モデル | 重複なしLogKV、2層、d512/H8/ff1024/C4 |
| 標準構成 | 各Block入力に幅4のConv残差、phaseなし、gate/self slotあり、固定−i log C |
| 追加方式 | RoPE、相対K/V、圧縮位置変換、KV/V norm、増幅は使用しない |
| パラメータ数 | 5,792,256 |
| 学習桁数M | {10,16,32,64}から一様選択 |
| 待ち時間T | loguniform[1,2028] |
| 先頭空白P | 整数[0,63]から一様選択 |
| 系列長 | P+T+2M、最大2,219 |
| 学習 | 各50k steps、実効batch64、microbatch32×2回蓄積、microbatch間でM/T/P共有 |
| 最適化 | AdamW lr3e-4、warmup1000、weight decay0、clip1 |
| 損失 | 全位置のposition-aligned CE、正解を入力へ戻さない |
| seed | 初期化0、内容生成1、M/T/P生成2、検証54321、最終評価12345 |
| 学習・評価精度 | fp32重み＋bf16 autocast（以前の可変桁実験と同じ） |
| best選択 | 2000 stepsごとの独立検証集合、macro完全一致率→macro桁精度→−訓練EMA |

タスク生成器は以前の`24b360c`からバイト単位で同一。訓練・検証・評価の手順と
乱数消費順を維持し、初期化のみ標準CausalConvへ変更した。旧モデルと共通する全重みの
初期値は以前と同一（SHA256 `2898927c4fec087b5f0d10ab5886d8bc40184c79bdff06bed551da9fe73369b5`）。
追加Convパラメータだけをseed0で新規初期化し、両タスクで同一の未学習モデルから始める。
固定10桁の学習済み重みからの転移ではない。

以前は厳密な決定論モードを無効にした実行を含むが、今回は有効にする。
旧6方式の完了済み結果と比較し、旧方式は再学習しない。共通初期値・データ条件は揃えるが、
決定論モードの差と各1 seedという限界を含む過去実験との比較として扱う。

固定10桁実験とはM/P以外にも、勾配蓄積、best選択、評価時の重み精度が異なる。
精度差を記憶桁数だけの効果とは断定しない。

## 評価

best選択用の検証はM4種×T{16,64,256,1024}×P{0,7}、各32例。
最終評価とはseedを分離する。最終評価はbest/finalそれぞれ以下の220セル・各256例。

- M4種×T=1〜131,072の従来41点、P=0：164セル
- M4種×T{16,64,256,2048}×P{7,15,63}：48セル
- 未学習M128×T{16,64,256,2048}×P{0,15}：8セル

合計2タスク×2checkpoint×220＝880セル。M64は学習範囲内、M128は桁数の外挿である。
P7/15/63は学習範囲内。chunk4096で全回答部分を回収し、回答がchunk境界をまたいでも採点する。
セル別seedをtask/M/T/Pから決めるため、評価順序は入力を変えない。
各例の記憶列、実配置、予測、marginをRAID上の圧縮NPZへ保存し、完了時にCPUで
元生成器から再生して照合する。リポジトリには指標・ログ・桁別誤答・NPZハッシュを収録する。

## 実行・再現

- 実験API：`exp/variable_memory/`、commit `90f13b1`。
- [実行コード・記録](experiments/logkv-variable-memory-20260914/README.md)
- [凍結33ファイルのハッシュ](experiments/logkv-variable-memory-20260914/source_manifest.json)
- 保存先：`/mnt/raid0/RecursiveCompressor/experiments/logkv-variable-memory-20260914/`
- 可変タスク・固定10桁タスクとの一致・回答chunk跨ぎ・評価seedの14 testsを通過。
- GPU0=Copying、GPU1=Selective。事前最大M64/P63/T2028のmicrobatch32×2更新後、
  本物の訓練300 stepsと220セル×16例の評価を両GPUで測定する。
- 8時間以上の見積もりなら本学習前に確認。実行上限は事前GPU確認開始から7.5時間。
  失敗時は両workerを停止し、自動再試行や次のcampaignは行わない。
- 追加seed、別幅・層数、可変桁16M評価は今回の範囲に含めない。
