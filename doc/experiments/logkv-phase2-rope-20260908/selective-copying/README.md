# selective-copying段階の評価結果

CPU自動集計。2 GPU以内の継続は承認済み。16M評価・追加アブレーションは開始しない。

[条件と実装](../../../logkv-phase2-rope.md)。各方式は固定M=10、50,000 steps、各T256例、1学習seed。
全33評価点は[metrics.csv](metrics.csv)、実行情報と検証結果は[summary.json](summary.json)。

| 方式 | best step | 学習時間（分） | 最初に完全一致率100%を下回る評価T best / final |
|---|---:|---:|---|
| phase2 | 49000 | 135.74 | 2 / 2 |
| retrieval-rope | 48100 | 227.78 | 2 / 3 |
| compressor-rope | 49000 | 141.65 | 2 / 2 |

| 方式 | checkpoint | T64 | T1024 | T2048 | T4096 | T8192 |
|---|---|---:|---:|---:|---:|---:|
| phase2 | best | 24/256 | 4/256 | 3/256 | 1/256 | 2/256 |
| phase2 | final | 31/256 | 1/256 | 2/256 | 3/256 | 1/256 |
| retrieval-rope | best | 69/256 | 3/256 | 3/256 | 5/256 | 0/256 |
| retrieval-rope | final | 25/256 | 0/256 | 0/256 | 0/256 | 0/256 |
| compressor-rope | best | 59/256 | 7/256 | 0/256 | 1/256 | 0/256 |
| compressor-rope | final | 53/256 | 3/256 | 0/256 | 0/256 | 0/256 |

数値は10桁すべて完全一致した例数。評価点間のすべてのTでの成功や16M保持を保証しない。

![評価](comparison.png)

![学習曲線](learning.png)
