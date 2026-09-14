# Main architecture without phase embeddings: automatically audited results

One training seed; best selected by training intervals. Human interpretation pending.

| mode | checkpoint | T | exact / 256 | digit accuracy |
|---|---|---:|---:|---:|
| copying | best | 3 | 226 | 0.988281 |
| copying | best | 16 | 114 | 0.937891 |
| copying | best | 64 | 31 | 0.861328 |
| copying | best | 128 | 71 | 0.893750 |
| copying | best | 256 | 1 | 0.750000 |
| copying | best | 1024 | 0 | 0.735156 |
| copying | best | 8192 | 0 | 0.721875 |
| copying | best | 131072 | 1 | 0.719141 |
| copying | final | 3 | 233 | 0.990234 |
| copying | final | 16 | 145 | 0.948047 |
| copying | final | 64 | 5 | 0.801172 |
| copying | final | 128 | 2 | 0.820312 |
| copying | final | 256 | 0 | 0.758594 |
| copying | final | 1024 | 1 | 0.722266 |
| copying | final | 8192 | 0 | 0.723047 |
| copying | final | 131072 | 1 | 0.709766 |
| selective-copying | best | 3 | 68 | 0.872656 |
| selective-copying | best | 16 | 9 | 0.718750 |
| selective-copying | best | 64 | 0 | 0.606641 |
| selective-copying | best | 128 | 2 | 0.631250 |
| selective-copying | best | 256 | 0 | 0.611719 |
| selective-copying | best | 1024 | 1 | 0.564844 |
| selective-copying | best | 8192 | 0 | 0.445703 |
| selective-copying | best | 131072 | 0 | 0.217188 |
| selective-copying | final | 3 | 91 | 0.895703 |
| selective-copying | final | 16 | 10 | 0.747266 |
| selective-copying | final | 64 | 1 | 0.557031 |
| selective-copying | final | 128 | 0 | 0.609766 |
| selective-copying | final | 256 | 0 | 0.494531 |
| selective-copying | final | 1024 | 0 | 0.440234 |
| selective-copying | final | 8192 | 0 | 0.378516 |
| selective-copying | final | 131072 | 0 | 0.178906 |
