# Two-layer LogKV with causal convolution and no phase embeddings: automatically audited results

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
| copying | final | 16 | 256 | 1.000000 |
| copying | final | 64 | 256 | 1.000000 |
| copying | final | 128 | 256 | 1.000000 |
| copying | final | 256 | 256 | 1.000000 |
| copying | final | 1024 | 256 | 1.000000 |
| copying | final | 8192 | 256 | 1.000000 |
| copying | final | 131072 | 256 | 1.000000 |
| selective-copying | best | 3 | 256 | 1.000000 |
| selective-copying | best | 16 | 235 | 0.990625 |
| selective-copying | best | 64 | 94 | 0.903906 |
| selective-copying | best | 128 | 84 | 0.892188 |
| selective-copying | best | 256 | 29 | 0.812500 |
| selective-copying | best | 1024 | 10 | 0.771484 |
| selective-copying | best | 8192 | 3 | 0.683984 |
| selective-copying | best | 131072 | 0 | 0.410547 |
| selective-copying | final | 3 | 255 | 0.999609 |
| selective-copying | final | 16 | 239 | 0.992969 |
| selective-copying | final | 64 | 112 | 0.908203 |
| selective-copying | final | 128 | 95 | 0.895703 |
| selective-copying | final | 256 | 35 | 0.826172 |
| selective-copying | final | 1024 | 19 | 0.785547 |
| selective-copying | final | 8192 | 11 | 0.714844 |
| selective-copying | final | 131072 | 0 | 0.482422 |
