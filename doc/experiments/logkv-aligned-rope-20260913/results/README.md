# End-anchored RoPE comparison: automatically audited results

One training seed; best selected by training intervals. Human interpretation pending.

| mode | checkpoint | T | exact / 256 | digit accuracy |
|---|---|---:|---:|---:|
| local-control | best | 3 | 248 | 0.996875 |
| local-control | best | 16 | 253 | 0.998828 |
| local-control | best | 8192 | 252 | 0.998437 |
| local-control | best | 32768 | 254 | 0.999219 |
| local-control | best | 49152 | 254 | 0.999219 |
| local-control | best | 65536 | 253 | 0.998828 |
| local-control | best | 98304 | 254 | 0.999219 |
| local-control | best | 131072 | 252 | 0.998437 |
| local-control | final | 3 | 256 | 1.000000 |
| local-control | final | 16 | 255 | 0.999609 |
| local-control | final | 8192 | 250 | 0.997656 |
| local-control | final | 32768 | 250 | 0.997656 |
| local-control | final | 49152 | 252 | 0.998437 |
| local-control | final | 65536 | 235 | 0.991797 |
| local-control | final | 98304 | 240 | 0.992969 |
| local-control | final | 131072 | 213 | 0.982422 |
| aligned | best | 3 | 10 | 0.828906 |
| aligned | best | 16 | 256 | 1.000000 |
| aligned | best | 8192 | 0 | 0.479297 |
| aligned | best | 32768 | 0 | 0.139063 |
| aligned | best | 49152 | 0 | 0.135547 |
| aligned | best | 65536 | 0 | 0.126562 |
| aligned | best | 98304 | 0 | 0.135937 |
| aligned | best | 131072 | 0 | 0.120703 |
| aligned | final | 3 | 28 | 0.910937 |
| aligned | final | 16 | 113 | 0.943359 |
| aligned | final | 8192 | 1 | 0.626953 |
| aligned | final | 32768 | 0 | 0.131641 |
| aligned | final | 49152 | 0 | 0.116406 |
| aligned | final | 65536 | 0 | 0.115234 |
| aligned | final | 98304 | 0 | 0.128906 |
| aligned | final | 131072 | 0 | 0.128125 |
