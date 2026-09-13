# 読み出し介入診断

2026-09-10 16:11 JSTに完了。本評価＋自動集計は約36秒（準備・事前検証時間を除く）。
[結果](results/README.md)、[照合記録](results/review.json)。全GPU解放済み。

[実験計画と制約](../../logkv-readout-intervention.md)。最大3GPU、既存モデルのみ、上限2時間。

- `common.py`：保存先・凍結source・3workerの割当
- `probe.py`：1回答位置への介入と観測、共通prefix再利用
- `preflight.py`：CPU独立oracle、GPU bf16の計測・sham一致
- `worker.py`：既存16例と別seed32例、対応するT間での介入と逆方向対照
- `run.py`：3workerを並列実行し、失敗・期限で停止
- `summarize.py`：CPUで予測・logit差・介入・不変条件を照合、結果と図を保存

実験rootは`/mnt/raid0/RecursiveCompressor/experiments/logkv-readout-intervention-20260910`。
`campaign.json`が全体状態、`{best-bf16,final-bf16,best-fp32}.log`がworkerログ。
完了時には本ディレクトリの`results/`へ集計・各例のlogit/targetを保存する。
baselineの中間表現も保存し、全介入後の中間表現はRAID上の場所とSHA256を索引に残す。

対照Tの中間表現を使う診断であり、修正版モデルの性能評価ではない。自動で追加実験を起動しない。

`review.py`は全350条件とRAID上の全中間表現のSHA256を再照合し、介入間のlogit一致も確認する。
