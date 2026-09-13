# phase2 + local RoPE

2026-09-09、両タスクの全6 runと396評価セルを完了・確認済み。GPU解放済み。
[Copying結果](copying/README.md)、[Selective結果](selective-copying/README.md)、
[確認記録](review.json)を保存した。以下の実行手順は完了した段階の再現用。

[設計・条件・結果](../../logkv-phase2-rope.md)。
`benchmark.py`は単一GPUで最大Tのforward/backwardと同じ200学習step相当の時間を測る。
`eval_preflight.py`は標準評価の最大級バッチでVRAMを確認する。未学習モデルの精度は性能評価ではない。
`benchmark.json`の初期重みSHA256はパラメータを順番にfloat32バイト列として連結した値。
`tests.log`は実装後の全LogKV/LMテスト出力。

`run_stage.py --task copying`はCopyingだけを3方式、各50k steps、best/final各33点×256例で比較する。
GPU 0でphase2→compressor-rope、GPU 1でretrieval-rope。最大2 GPU、段階全体で8時間の上限。
追加のユーザー指示で、2 GPU以内の連続実験が承認された。
`continue_campaign.py`がCopyingの完了・CPU検証を待って、別タスクのSelectiveを起動する。
その所要時間はCopyingのGPUごとの実測合計の最大値×1.25＋0.25時間で保守的に推定する。
推定8時間以上、Copyingの異常終了・検証失敗、既存Selective段階がある場合は自動起動しない。
2段階の合計は約12〜14時間を見込む。Selective終了後はCPU集計まで行って停止する。

実験root：`/mnt/raid0/RecursiveCompressor/experiments/logkv-phase2-rope-20260908`。
その`source/`に実装commitを凍結したworktreeを置き、`exp/{task}/{mode}-fixed10-20260908/`にcheckpointを保存する。
タスク名のディレクトリへmanifestとtrain/evalログを保存する。既存段階やrunへの上書きは拒否する。
再実行時はスクリプト内のROOT/SOURCEを新しい場所へ変更する。

リポジトリrootからの開始コマンド（この段階はCopyingのみ）:

```bash
.venv/bin/python doc/experiments/logkv-phase2-rope-20260908/run_stage.py --task copying
```

モデル実装の既定値は従来と同一。実験では`--phase-emb --phase-levels 2 --gated-attention --self-slot`に加えて
`--retrieval-rope`または`--compressor-rope`を片方だけ指定する。phase2対照はどちらも指定しない。

`progress.py`はCopying段階の読取専用ステータス表示。
`summarize_stage.py --task copying`は段階完了後だけ実行できるCPU集計で、
全runの終了コード・設定差分・best選択・評価点・正解数の整合性を確認し、生ログと結果を保存する。

`finish_task.py --task ...`は完了した単一タスクのCPU集計・結果表保存のみを実行する。
旧`finish_copying.py`の待機プロセスは停止済みで、継続コーディネータに置き換えた。
継続状態は実験rootの`continuation.json`に記録し、終了/停止時に本ディレクトリへも保存する。
コーディネータはファイルロックで二重起動を防ぎ、16M評価・追加アブレーション・commit/mergeを行わない。
