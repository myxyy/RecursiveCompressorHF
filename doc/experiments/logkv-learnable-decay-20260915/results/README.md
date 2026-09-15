# Two-layer CausalConv LogKV with learnable level decay: automatically audited results

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
| copying | final | 3 | 35 | 0.910937 |
| copying | final | 16 | 256 | 1.000000 |
| copying | final | 64 | 256 | 1.000000 |
| copying | final | 128 | 256 | 1.000000 |
| copying | final | 256 | 256 | 1.000000 |
| copying | final | 1024 | 255 | 0.999609 |
| copying | final | 8192 | 150 | 0.958594 |
| copying | final | 131072 | 75 | 0.928906 |
| selective-copying | best | 3 | 256 | 1.000000 |
| selective-copying | best | 16 | 256 | 1.000000 |
| selective-copying | best | 64 | 248 | 0.996875 |
| selective-copying | best | 128 | 255 | 0.999609 |
| selective-copying | best | 256 | 237 | 0.989844 |
| selective-copying | best | 1024 | 190 | 0.966797 |
| selective-copying | best | 8192 | 8 | 0.675000 |
| selective-copying | best | 131072 | 0 | 0.464844 |
| selective-copying | final | 3 | 255 | 0.999609 |
| selective-copying | final | 16 | 196 | 0.972656 |
| selective-copying | final | 64 | 210 | 0.979297 |
| selective-copying | final | 128 | 209 | 0.979688 |
| selective-copying | final | 256 | 175 | 0.960938 |
| selective-copying | final | 1024 | 85 | 0.904687 |
| selective-copying | final | 8192 | 3 | 0.671094 |
| selective-copying | final | 131072 | 0 | 0.444922 |
