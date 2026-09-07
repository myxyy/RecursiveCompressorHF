# 可変桁数・階層位置表現の比較データ

方式・実装・評価上の制約は[実験レポート](../../logkv-position-study.md)を参照。
実験コードの固定commitは `24b360c`。各方式を1台のGPUに割り当て、Copying / Selective Copyingを各50,000 steps学習した。

- `runs/<mode>-<task>/run_config.json`: 学習条件・共通初期重みのSHA256
- `train_log.jsonl`: 100 stepsごとの訓練指標、2,000 stepsごとの独立した検証集合の値
- `best.json`: 検証macro string → macro token → −訓練EMAの辞書順で選んだcheckpoint
- `model_config_best.json` / `model_config_final.json`: 保存モデルの設定
- `results_best.json` / `results_final.json`: 各220セル、各256例の最終評価。正解数も保存
- `position_stats_best.json` / `position_stats_final.json`: 学習後の反射ベクトル・直交性・変換順序の差・相対K/Vノルム
- `export.py`: 元の実行ディレクトリから記録を集め、checkpointから位置パラメータ統計を抽出するコード
- `<mode>-<task>.json`: GPU番号、固定source、commit、実行コマンド、開始終了時刻と終了コード
- `comparison.csv`: 全5,280評価行
- `summary.json`: best選択結果、M別の完全一致した待ち時間数と最小正解率
- `tables.md` / `prefix-and-unseen-memory.md`: 代表待ち時間、prefix、未学習M128の一覧
- `*.png`: 待ち時間別の正解率と学習曲線
- `probe.json` / `probe-reflections-only.json`: 学習前の合成線形符号診断。LMのCopying性能ではない
- `trained-encoder-diagnostics.md`: 学習済み位置変換の全72件の事後診断表
- `trained-encoder-*.json` / `probe_trained.py`: 学習済み圧縮変換の事後診断。3方式×best/final×2層×M{4,16,64}、各256例。均一な圧縮重みと外付け線形復号器を使い、LMの読み出し・内容依存の重みは検証しない。M記号から1要約を作る条件であり、待ち時間Tの追加再圧縮は含まない
- `smoke-*.json` / `compatibility.json`: GPU学習smokeとデフォルト無効時の互換性検証
- `environment.json`: 実験環境

`probe*.json`のrank不足の条件数にはPython JSON表現の `Infinity` を含む。
これは合成符号行列の特異性を表し、学習や最終評価のNaN/Infを表すものではない。

## 集計の再現

このディレクトリの `summarize.py` は実験モデルをimportせず、Pythonとmatplotlibで実行できる。
main上でも全runの完了、条件の一致、best選択、評価セル・正解数を検査して表と図を再生成する。

```bash
uv run python doc/experiments/logkv-position-study-20260907/summarize.py
```

同梱の `probe.py` / `smoke.py` は実行時のコードを保存したもので、実験ブランチのモデルと
`exp.position_study` に依存する。再実行する場合はcommit `24b360c` のworktreeで、
`exp/position_study/probe.py` / `exp/position_study/smoke.py` を実行する。

モデル重み・optimizer状態はgitに含めず、実行manifestの `run_dir` に保存する。
本評価の乱数seedは12345、checkpoint選択は54321、学習はseed0の1回である。

位置パラメータ統計を含めて元のcheckpointからアーカイブを再作成するには、保存された実行ディレクトリを指定する。
この処理にはPyTorch、safetensorsと元の重み・固定sourceが必要になる。

```bash
uv run python doc/experiments/logkv-position-study-20260907/export.py \
  --root /mnt/raid0/RecursiveCompressor/experiments/logkv-position-study-20260907 \
  --dest /tmp/logkv-position-study-archive
```

学習済み圧縮変換の事後診断は、学習終了後のcheckpointに対して次で再現できる。
Selective Copyingは `--task selective-copying` に変更する。

```bash
uv run python doc/experiments/logkv-position-study-20260907/probe_trained.py \
  --root /mnt/raid0/RecursiveCompressor/experiments/logkv-position-study-20260907 \
  --source /path/to/worktree-at-24b360c --task copying \
  --output /tmp/trained-encoder-copying.json
```
