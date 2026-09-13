# End-anchored RoPE Selective Copying: automatically audited results

One training seed; best selected by training intervals. Human interpretation pending.

| mode | checkpoint | T | exact / 256 | digit accuracy |
|---|---|---:|---:|---:|
| local-control | best | 3 | 216 | 0.981641 |
| local-control | best | 16 | 107 | 0.912891 |
| local-control | best | 64 | 17 | 0.760547 |
| local-control | best | 128 | 21 | 0.760938 |
| local-control | best | 256 | 3 | 0.660937 |
| local-control | best | 512 | 6 | 0.655859 |
| local-control | best | 1024 | 1 | 0.590625 |
| local-control | best | 2048 | 1 | 0.569922 |
| local-control | best | 4096 | 0 | 0.476172 |
| local-control | best | 8192 | 0 | 0.462891 |
| local-control | final | 3 | 232 | 0.988672 |
| local-control | final | 16 | 99 | 0.904687 |
| local-control | final | 64 | 28 | 0.800000 |
| local-control | final | 128 | 14 | 0.773828 |
| local-control | final | 256 | 1 | 0.683203 |
| local-control | final | 512 | 3 | 0.669922 |
| local-control | final | 1024 | 0 | 0.647656 |
| local-control | final | 2048 | 2 | 0.619922 |
| local-control | final | 4096 | 0 | 0.566406 |
| local-control | final | 8192 | 0 | 0.536328 |
| aligned | best | 3 | 230 | 0.989453 |
| aligned | best | 16 | 79 | 0.894922 |
| aligned | best | 64 | 25 | 0.799609 |
| aligned | best | 128 | 10 | 0.751172 |
| aligned | best | 256 | 2 | 0.665625 |
| aligned | best | 512 | 1 | 0.628906 |
| aligned | best | 1024 | 0 | 0.525391 |
| aligned | best | 2048 | 0 | 0.448047 |
| aligned | best | 4096 | 0 | 0.210156 |
| aligned | best | 8192 | 0 | 0.193359 |
| aligned | final | 3 | 212 | 0.980469 |
| aligned | final | 16 | 108 | 0.922656 |
| aligned | final | 64 | 17 | 0.791406 |
| aligned | final | 128 | 11 | 0.748828 |
| aligned | final | 256 | 1 | 0.670312 |
| aligned | final | 512 | 2 | 0.647656 |
| aligned | final | 1024 | 0 | 0.546484 |
| aligned | final | 2048 | 0 | 0.448047 |
| aligned | final | 4096 | 0 | 0.200781 |
| aligned | final | 8192 | 0 | 0.178906 |
