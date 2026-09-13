# 位置表現実験コードの保存

`experiment-code.tar.gz`は実験ブランチのコードを元のパス・内容で保存したアーカイブ。
元commitは`537e538`、各ファイルのSHA256は`manifest.json`を参照。
可変桁数実験の`exp/position_study/`、実験用Copying CLI、位置機能を含むテストを収録する。
`CLAUDE-experimental.md`は実験ブランチ時点の履歴資料で、mainの実行指示ではない。

これらのコードにはmainに存在しない位置表現のAPIが必要。
mainの訓練・通常テストに混入しないよう、元ファイルを改変せずアーカイブ化した。
実行には各レポートに指定したcommitの別worktreeを使う。
可変桁数の原実験は`24b360c`、RoPEの最終実装は`8dcfea4`。
新しく全実験を起動する指示ではなく、再現時のソース指定である。

```bash
tar -tzf doc/experiments/position-code-archive-20260913/experiment-code.tar.gz
# 例：別の作業ディレクトリで原実験の実装を参照
git worktree add --detach /mnt/raid0/RecursiveCompressor/position-study-review 24b360c
```

モデル本体・設定・通常の訓練CLIはmainへ取り込んでいない。
実験ブランチ`logkv-aligned-rope`もそのまま保存している。
