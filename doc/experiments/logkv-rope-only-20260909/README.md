# RoPEのみ（phase2なし）

2026-09-10、[Copyingの結果](copying/README.md)と[確認記録](copying/review.json)を保存した。
[Selectiveの結果](selective-copying/README.md)と[確認記録](selective-copying/review.json)も保存した。
全4 runは04:52 JSTまでに完了し、約7時間36分で8時間上限内。全GPU解放済み、追加実験は未起動。

[条件・状態・結果](../../logkv-rope-only.md)。

- `preflight.py/json`: CPUでphase無効化、初期重み、パラメータ数、有限出力を確認
- `run.py`: 2方式×2タスクを、最大2 GPU・全体8時間上限で実行
- `summarize_stage.py`: 完了した単一タスクのログ・config・best選択・結果をCPUで検証/保存
- `finish_task.py`: 単一タスクの結果表とレポート状態を更新
- `compare_phase.py`: 前回のphase2あり結果をコピー・比較してCSV/図/表を保存

実験rootは`/mnt/raid0/RecursiveCompressor/experiments/logkv-rope-only-20260909`。
前回の凍結source `/mnt/raid0/RecursiveCompressor/experiments/logkv-phase2-rope-20260908/source`を再利用する。
今回の学習・checkpointは`exp/{task}/{mode}-no-phase-fixed10-20260909/`へ保存し、前回の結果は上書きしない。

```bash
.venv/bin/python doc/experiments/logkv-rope-only-20260909/run.py
```

GPU 1で読み出しRoPEのみ、GPU 0でCompressor RoPEのみ。
Copying完了・検証後にSelectiveを別々に学習・評価する。
全体で8時間を超えると予測した場合は、Selective開始前に確認待ちで停止する。
各runのmanifestとログは実験rootのタスク名ディレクトリ、全体状態は`campaign.json`。
ファイルロック・既存campaign/run拒否により二重起動を防ぐ。再実行は新ROOTで行い、自動再試行しない。
GPU実験終了後はCPU集計まで実行し、16M・追加アブレーション・commit/mergeは自動実行しない。

前回と同じseed番号だが、phase除去で共通重みの初期値も変わる。この差を`preflight.json`に記録している。

## T131072追加評価

2026-09-10 01:47 JSTに完了。[結果・図表](extension-131072/README.md)を保存し、GPU 2は解放済み。
元のSelective学習もその後04:52 JSTに完了した。

`extend_131072.py`は明示承認済みのGPU 2で読み出しRoPEのCopying best/finalを各41点×256例で評価する。
元のcheckpointへの読取参照と独立したDATA_DIRを使用し、元結果を上書きしない。追加評価の上限は1時間。
`summarize_extension.py`はCPUで全82セル・重複する各33点の一致を確認し、結果と図を`extension-131072/`へ保存する。
元のSelective学習とその8時間上限は変更しない。この追加評価へのGPU許可は通常の2 GPU上限とは別の個別承認。
