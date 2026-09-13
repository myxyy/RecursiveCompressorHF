# 自己スロット除去 Copying 実験

2026-09-10 10:30:06〜14:35:35 JST、約4時間6分で正常終了。GPU解放済み。
[結果](results/README.md)、[照合記録](results/review.json)、[開始記録](started.json)。

[目的・条件・診断範囲](../../logkv-self-slot.md)。1GPU、単一の50k学習と評価・診断、全体7.5時間上限。

- `common.py`：保存先と凍結source
- `preflight.py/json`：同一初期重み/RNG、自己スロットなしの独立oracle・勾配・streaming検証
- `evaluate_detail.py`：凍結標準評価器に各例の予測・logit差の保存を追加
- `diagnose.py`：固定16記憶列・注意量・fp32比較・既存9誤答の再現。`--validate-only`はCPU事前検証
- `run.py`：GPU 0で順次実行、timeout/失敗時停止、二重起動拒否
- `summarize.py`：CPUで結果照合、対照との比較表・図・raw記録を`results/`へ保存

実験root：`/mnt/raid0/RecursiveCompressor/experiments/logkv-self-slot-20260910`。
状態：実験rootの`campaign.json`、ログ：`train.log` / `best.log` / `final.log` / `diagnose.log`。
完了・停止時は本ディレクトリにも`campaign.json`を保存する。
モデルは実験rootの`exp/copying/retrieval-rope-no-phase-no-self-fixed10-20260910/`へ保存する。

既存の自己スロットありモデルは読み取り専用の対照として利用し、元結果を上書きしない。
標準評価は各41点・256例、診断は基本各16例。fp32の誤答再評価は選別された例なので標準精度と混同しない。
追加学習・16M・Selective・git commit/mergeはランナーから実行しない。

`review_results.py`はCPUのみでアーカイブ・logit差・対応する記憶列を照合し、診断結果と図を集計する。
