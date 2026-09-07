# 学習済み位置変換の事後診断

均一な圧縮重み・固定乱数の8記号ベクトル・外付け線形復号器、各256例。LMの読み出し精度ではない。bf16化は最終要約だけ。

## copying

| mode | checkpoint | layer | M | rank / features | condition number | fp64 string (%) | bf16 summary string (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| binding | best | 0 | 4 | 32 / 32 | 1.56e+00 | 100.00 | 100.00 |
| binding | best | 0 | 16 | 128 / 128 | 2.91e+00 | 100.00 | 100.00 |
| binding | best | 0 | 64 | 512 / 512 | 1.40e+03 | 100.00 | 100.00 |
| binding | best | 1 | 4 | 32 / 32 | 1.56e+00 | 100.00 | 100.00 |
| binding | best | 1 | 16 | 128 / 128 | 2.93e+00 | 100.00 | 100.00 |
| binding | best | 1 | 64 | 512 / 512 | 2.33e+03 | 100.00 | 100.00 |
| binding | final | 0 | 4 | 32 / 32 | 1.56e+00 | 100.00 | 100.00 |
| binding | final | 0 | 16 | 128 / 128 | 2.91e+00 | 100.00 | 100.00 |
| binding | final | 0 | 64 | 512 / 512 | 1.81e+03 | 100.00 | 100.00 |
| binding | final | 1 | 4 | 32 / 32 | 1.56e+00 | 100.00 | 100.00 |
| binding | final | 1 | 16 | 128 / 128 | 2.93e+00 | 100.00 | 100.00 |
| binding | final | 1 | 64 | 512 / 512 | 4.16e+03 | 100.00 | 100.00 |
| combined | best | 0 | 4 | 32 / 32 | 1.55e+00 | 100.00 | 100.00 |
| combined | best | 0 | 16 | 128 / 128 | 2.87e+00 | 100.00 | 100.00 |
| combined | best | 0 | 64 | 512 / 512 | 1.54e+04 | 100.00 | 78.12 |
| combined | best | 1 | 4 | 32 / 32 | 1.56e+00 | 100.00 | 100.00 |
| combined | best | 1 | 16 | 128 / 128 | 2.89e+00 | 100.00 | 100.00 |
| combined | best | 1 | 64 | 512 / 512 | 1.06e+03 | 100.00 | 100.00 |
| combined | final | 0 | 4 | 32 / 32 | 1.55e+00 | 100.00 | 100.00 |
| combined | final | 0 | 16 | 128 / 128 | 2.87e+00 | 100.00 | 100.00 |
| combined | final | 0 | 64 | 512 / 512 | 8.54e+03 | 100.00 | 98.44 |
| combined | final | 1 | 4 | 32 / 32 | 1.56e+00 | 100.00 | 100.00 |
| combined | final | 1 | 16 | 128 / 128 | 2.89e+00 | 100.00 | 100.00 |
| combined | final | 1 | 64 | 512 / 512 | 1.28e+03 | 100.00 | 100.00 |
| combined-no-decay | best | 0 | 4 | 32 / 32 | 1.55e+00 | 100.00 | 100.00 |
| combined-no-decay | best | 0 | 16 | 128 / 128 | 2.88e+00 | 100.00 | 100.00 |
| combined-no-decay | best | 0 | 64 | 512 / 512 | 9.05e+02 | 100.00 | 100.00 |
| combined-no-decay | best | 1 | 4 | 32 / 32 | 1.56e+00 | 100.00 | 100.00 |
| combined-no-decay | best | 1 | 16 | 128 / 128 | 2.93e+00 | 100.00 | 100.00 |
| combined-no-decay | best | 1 | 64 | 512 / 512 | 8.25e+02 | 100.00 | 100.00 |
| combined-no-decay | final | 0 | 4 | 32 / 32 | 1.55e+00 | 100.00 | 100.00 |
| combined-no-decay | final | 0 | 16 | 128 / 128 | 2.88e+00 | 100.00 | 100.00 |
| combined-no-decay | final | 0 | 64 | 512 / 512 | 9.05e+02 | 100.00 | 100.00 |
| combined-no-decay | final | 1 | 4 | 32 / 32 | 1.56e+00 | 100.00 | 100.00 |
| combined-no-decay | final | 1 | 16 | 128 / 128 | 2.93e+00 | 100.00 | 100.00 |
| combined-no-decay | final | 1 | 64 | 512 / 512 | 8.25e+02 | 100.00 | 100.00 |

## selective-copying

| mode | checkpoint | layer | M | rank / features | condition number | fp64 string (%) | bf16 summary string (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| binding | best | 0 | 4 | 32 / 32 | 1.55e+00 | 100.00 | 100.00 |
| binding | best | 0 | 16 | 128 / 128 | 2.93e+00 | 100.00 | 100.00 |
| binding | best | 0 | 64 | 512 / 512 | 1.61e+03 | 100.00 | 100.00 |
| binding | best | 1 | 4 | 32 / 32 | 1.56e+00 | 100.00 | 100.00 |
| binding | best | 1 | 16 | 128 / 128 | 2.88e+00 | 100.00 | 100.00 |
| binding | best | 1 | 64 | 512 / 512 | 3.33e+03 | 100.00 | 100.00 |
| binding | final | 0 | 4 | 32 / 32 | 1.55e+00 | 100.00 | 100.00 |
| binding | final | 0 | 16 | 128 / 128 | 2.93e+00 | 100.00 | 100.00 |
| binding | final | 0 | 64 | 512 / 512 | 1.61e+03 | 100.00 | 100.00 |
| binding | final | 1 | 4 | 32 / 32 | 1.56e+00 | 100.00 | 100.00 |
| binding | final | 1 | 16 | 128 / 128 | 2.88e+00 | 100.00 | 100.00 |
| binding | final | 1 | 64 | 512 / 512 | 3.33e+03 | 100.00 | 100.00 |
| combined | best | 0 | 4 | 32 / 32 | 1.56e+00 | 100.00 | 100.00 |
| combined | best | 0 | 16 | 128 / 128 | 2.88e+00 | 100.00 | 100.00 |
| combined | best | 0 | 64 | 512 / 512 | 5.45e+03 | 100.00 | 100.00 |
| combined | best | 1 | 4 | 32 / 32 | 1.55e+00 | 100.00 | 100.00 |
| combined | best | 1 | 16 | 128 / 128 | 2.90e+00 | 100.00 | 100.00 |
| combined | best | 1 | 64 | 512 / 512 | 1.79e+03 | 100.00 | 100.00 |
| combined | final | 0 | 4 | 32 / 32 | 1.56e+00 | 100.00 | 100.00 |
| combined | final | 0 | 16 | 128 / 128 | 2.88e+00 | 100.00 | 100.00 |
| combined | final | 0 | 64 | 512 / 512 | 8.80e+03 | 100.00 | 94.53 |
| combined | final | 1 | 4 | 32 / 32 | 1.55e+00 | 100.00 | 100.00 |
| combined | final | 1 | 16 | 128 / 128 | 2.90e+00 | 100.00 | 100.00 |
| combined | final | 1 | 64 | 512 / 512 | 1.56e+03 | 100.00 | 100.00 |
| combined-no-decay | best | 0 | 4 | 32 / 32 | 1.56e+00 | 100.00 | 100.00 |
| combined-no-decay | best | 0 | 16 | 128 / 128 | 2.94e+00 | 100.00 | 100.00 |
| combined-no-decay | best | 0 | 64 | 512 / 512 | 5.00e+03 | 100.00 | 99.22 |
| combined-no-decay | best | 1 | 4 | 32 / 32 | 1.59e+00 | 100.00 | 100.00 |
| combined-no-decay | best | 1 | 16 | 128 / 128 | 2.89e+00 | 100.00 | 100.00 |
| combined-no-decay | best | 1 | 64 | 512 / 512 | 9.92e+02 | 100.00 | 100.00 |
| combined-no-decay | final | 0 | 4 | 32 / 32 | 1.56e+00 | 100.00 | 100.00 |
| combined-no-decay | final | 0 | 16 | 128 / 128 | 2.93e+00 | 100.00 | 100.00 |
| combined-no-decay | final | 0 | 64 | 512 / 512 | 9.05e+03 | 100.00 | 93.36 |
| combined-no-decay | final | 1 | 4 | 32 / 32 | 1.59e+00 | 100.00 | 100.00 |
| combined-no-decay | final | 1 | 16 | 128 / 128 | 2.89e+00 | 100.00 | 100.00 |
| combined-no-decay | final | 1 | 64 | 512 / 512 | 1.36e+04 | 100.00 | 80.08 |
