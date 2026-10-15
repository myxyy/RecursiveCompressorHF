# Two-layer LogKV with fixed zero level bias: automatically audited results

One training seed; best selected by training intervals. Human interpretation pending.

| mode | checkpoint | T | exact / 256 | digit accuracy |
|---|---|---:|---:|---:|
| copying | best | 3 | 256 | 1.000000 |
| copying | best | 16 | 256 | 1.000000 |
| copying | best | 64 | 256 | 1.000000 |
| copying | best | 128 | 256 | 1.000000 |
| copying | best | 256 | 256 | 1.000000 |
| copying | best | 1024 | 256 | 1.000000 |
| copying | best | 8192 | 256 | 1.000000 |
| copying | best | 131072 | 256 | 1.000000 |
| copying | final | 3 | 256 | 1.000000 |
| copying | final | 16 | 255 | 0.999609 |
| copying | final | 64 | 256 | 1.000000 |
| copying | final | 128 | 255 | 0.999609 |
| copying | final | 256 | 256 | 1.000000 |
| copying | final | 1024 | 255 | 0.999609 |
| copying | final | 8192 | 256 | 1.000000 |
| copying | final | 131072 | 256 | 1.000000 |
| selective-copying | best | 3 | 255 | 0.999609 |
| selective-copying | best | 16 | 244 | 0.994531 |
| selective-copying | best | 64 | 135 | 0.932031 |
| selective-copying | best | 128 | 128 | 0.918359 |
| selective-copying | best | 256 | 75 | 0.878125 |
| selective-copying | best | 1024 | 44 | 0.844922 |
| selective-copying | best | 8192 | 49 | 0.840234 |
| selective-copying | best | 131072 | 18 | 0.781250 |
| selective-copying | final | 3 | 256 | 1.000000 |
| selective-copying | final | 16 | 246 | 0.994141 |
| selective-copying | final | 64 | 121 | 0.918359 |
| selective-copying | final | 128 | 145 | 0.928516 |
| selective-copying | final | 256 | 71 | 0.865625 |
| selective-copying | final | 1024 | 45 | 0.840625 |
| selective-copying | final | 8192 | 43 | 0.831641 |
| selective-copying | final | 131072 | 12 | 0.750000 |
