# LogKV実験記録一覧

実験の成功・不成功を含めて記録を集約する。mainの標準実装と実験ブランチの実装は区別する。
各レポートに学習条件、評価条件、再現用commit、全評価値と成果物の場所を記載した。
モデル重みはDATA_DIRに保存し、リポジトリには設定・ログ・評価値・図・集計コードを収録する。

| 比較 | コード | 記録 |
|---|---|---|
| 過去のLM・記憶タスク検証 | 各節に記載 | [設計・実験記録](logkv.md) |
| 重複あり／重複なし参照 | `c9c51c9` / `3b0ce51` | [重複除去](logkv-refine-experiments.md) |
| phase2／retrieval相対logit補正／位置表現なし | `7261652` | [相対logit補正](logkv-relative-position-experiments.md) |
| Compressor固定／学習可能logit減衰 | `77356e6` | [圧縮減衰](logkv-compressor-experiments.md) |
| 子位置別直交変換／相対K/V／併用／レベル減衰除去 | `24b360c` (`logkv-position-study`) | [可変桁数比較](logkv-position-study.md) |

位置表現の実験APIは[別紙設計](logkv-position-design.md)を参照。
固定10桁の評価と可変桁数・prefix付きの評価は学習条件が異なり、数値を直接対照として扱わない。
