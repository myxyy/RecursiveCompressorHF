# 圧縮表現診断の成果物（2026-09-08）

[解釈・条件・全結果](../../logkv-compression-diagnostic.md)。固定10桁Copyingの学習済みfinalを診断し、再学習はしていない。
GPU 0のみで4 runとphase2のsource互換性確認を逐次実行。すべて完了し、追加実験はキューに入れていない。

## ファイル

- `probe.py`: 実際のprefix圧縮・blank対照・解答Query/logitを観測するGPUスクリプト
- `check_compatibility.py`: phase2の元source/現sourceで同一入力の語彙logitを保存
- `summarize.py`: CPUで正解数・保存値を検証し、CSV/JSON/図を生成、元runをコピー
- `runs/{mode}-{precision}/`: `metadata.json`、`trace.npz`、`run.log`
- `configs/`: 実際にロードしたcheckpointのconfig.jsonのコピー
- `performance.csv`: 28評価セル。T=N−9、n=16、M=10
- `representation.csv`: 8 headsを連結したK/V差分のノルム・cosine、112行
- `heads.csv`: head別指標896行。ゼロベクトルのcosineは空欄、有効数を保存
- `readout.csv`: head別の実Query/logit/attentionと固定Queryの内容差分スコア448行
- `first_child_transform.csv`: 保存パラメータからfloat64再構成したU0の幾何学的診断224行。モデル介入ではない
- `compat-{original,current}.npz`とログ、`compatibility.json`: source間logit比較
- `representation.png` / `performance.png`: 上記CSVと同じ集計から生成した図
- `environment.json`: 計測source・ツールバージョン
- `manifest.sha256`: 原記録・設定・スクリプトのチェックサム

NPZの`memory_l{0,1}_d{2..8}_{k,v,q}`は各層・段の先頭要約（16例×8 heads×64次元）。
`blank_`が同位置のblank対照、`_child`がpooling前の4子、`_weights`が実pooling重み、
`_pre_transform`が実際と同じ重みで位置変換前の子を混合した要約。
`_read_`は11 marker位置での実Query・検索指標（解答集計では先頭markerを除く）。
`memory_d{d}_answer_logits`は解答10位置の語彙logit、`digits`は正解列。
`parameters_`はcombinedの位置変換/相対Keyパラメータ。配列は保存時にfloat32へ変換している。

## 再現

リポジトリrootから実行する。スクリプト内のSOURCE/ROOT/CHECKPOINTSは本機の絶対パス。
実験rootは`/mnt/raid0/RecursiveCompressor/experiments/logkv-compression-diagnostic-20260908`。
GPU再実行は既存runディレクトリがあると拒否するため、再実行時は新しいROOTへ変更する。
checkpoint本体はGit成果物には含めず、パスとSHA256をmetadataに記録している。
以下は完了済みの段階の再現手順であり、次の段階の実行予約ではない。

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python doc/experiments/logkv-compression-diagnostic-20260908/probe.py --mode phase2 --precision bf16
CUDA_VISIBLE_DEVICES=0 .venv/bin/python doc/experiments/logkv-compression-diagnostic-20260908/probe.py --mode combined-no-decay --precision bf16
CUDA_VISIBLE_DEVICES=0 .venv/bin/python doc/experiments/logkv-compression-diagnostic-20260908/probe.py --mode phase2 --precision fp32
CUDA_VISIBLE_DEVICES=0 .venv/bin/python doc/experiments/logkv-compression-diagnostic-20260908/probe.py --mode combined-no-decay --precision fp32
```

各runの標準出力/標準エラーは実験rootの`{mode}-{precision}.log`へ保存する。
互換性確認は次の2呼び出しを逐次実行し、同様に`compat-original.log` / `compat-current.log`へ保存した。

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python doc/experiments/logkv-compression-diagnostic-20260908/check_compatibility.py \
  --source /mnt/raid0/RecursiveCompressor/experiments/logkv-refine-20260906/refined-source \
  --out /mnt/raid0/RecursiveCompressor/experiments/logkv-compression-diagnostic-20260908/compat-original.npz
CUDA_VISIBLE_DEVICES=0 .venv/bin/python doc/experiments/logkv-compression-diagnostic-20260908/check_compatibility.py \
  --source /mnt/raid0/RecursiveCompressor/experiments/logkv-position-study-20260907/source \
  --out /mnt/raid0/RecursiveCompressor/experiments/logkv-compression-diagnostic-20260908/compat-current.npz
.venv/bin/python doc/experiments/logkv-compression-diagnostic-20260908/summarize.py
```

集計のみならGPUは不要。別環境では`summarize.py`のROOTを変更し、保存済み`runs/`の各子ディレクトリをそのroot直下に置く。
`runs/{run}/run.log`はrootの`{run}.log`へ、`compat-*`のNPZとログもroot直下へ配置する。
