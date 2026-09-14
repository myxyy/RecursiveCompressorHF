# Three-layer main architecture without phase embeddings: automatically audited results

One training seed; best selected by training intervals. Human interpretation pending.

| mode | checkpoint | T | exact / 256 | digit accuracy |
|---|---|---:|---:|---:|
| copying | best | 3 | 255 | 0.999609 |
| copying | best | 16 | 238 | 0.992578 |
| copying | best | 64 | 237 | 0.992188 |
| copying | best | 128 | 225 | 0.987500 |
| copying | best | 256 | 0 | 0.559766 |
| copying | best | 1024 | 0 | 0.529687 |
| copying | best | 8192 | 0 | 0.527344 |
| copying | best | 131072 | 0 | 0.537500 |
| copying | final | 3 | 233 | 0.989844 |
| copying | final | 16 | 233 | 0.990625 |
| copying | final | 64 | 195 | 0.973828 |
| copying | final | 128 | 196 | 0.973047 |
| copying | final | 256 | 9 | 0.836328 |
| copying | final | 1024 | 0 | 0.424609 |
| copying | final | 8192 | 0 | 0.419922 |
| copying | final | 131072 | 0 | 0.417187 |
| selective-copying | best | 3 | 209 | 0.973437 |
| selective-copying | best | 16 | 27 | 0.830078 |
| selective-copying | best | 64 | 11 | 0.753516 |
| selective-copying | best | 128 | 18 | 0.781641 |
| selective-copying | best | 256 | 4 | 0.753125 |
| selective-copying | best | 1024 | 5 | 0.741016 |
| selective-copying | best | 8192 | 7 | 0.709375 |
| selective-copying | best | 131072 | 1 | 0.623828 |
| selective-copying | final | 3 | 197 | 0.971094 |
| selective-copying | final | 16 | 44 | 0.850781 |
| selective-copying | final | 64 | 16 | 0.769141 |
| selective-copying | final | 128 | 22 | 0.783594 |
| selective-copying | final | 256 | 0 | 0.653125 |
| selective-copying | final | 1024 | 1 | 0.641406 |
| selective-copying | final | 8192 | 1 | 0.622656 |
| selective-copying | final | 131072 | 0 | 0.573438 |
