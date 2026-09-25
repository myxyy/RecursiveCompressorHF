# ディレクトリ構成と旧パスからの移行

2026-09-25、LogKVと旧RecursiveCompressorのコードを別パッケージに整理した。
モデルの演算・パラメータ名・設定の既定値・チェックポイント形式は変更していない。
既存のHF形式のモデルディレクトリは、引き続き生成CLIの`--model-dir`に指定できる。

## 現行配置

```text
models/
  logkv/                  # attention.py / configuration.py / modeling.py / pipeline.py
  recursive_compressor/   # 旧アーキテクチャ。同じ役割でファイルを分割
training/                 # train_logkv.py / train_logkv_pipeline.py / train_pipeline.py
inference/                # predict.py / predict_logkv.py / predict_stream.py / chat_server.py
data_pipeline/            # dataset.py（共通データ処理）
benchmarks/               # benchmark_logkv_predict.py
tests/
  logkv/                  # LogKV、パイプライン、推論のテスト
  legacy/                 # RecursiveCompressor、共通データ処理のテスト
exp/
  copying/                # 固定10桁Copying
  selective_copying/      # 固定10桁Selective Copying
  variable_memory/        # 可変桁Copying / Selective Copying
doc/                      # 設計・実験レポート
  experiments/            # 当時のソース・結果・監査記録
```

ローカルの追跡対象外スクリプト`upload.py`は`tools/upload.py`へ移動した。
引き続きGitの追跡対象外であり、importではアップロードを実行しない。

## 起動方法

リポジトリルートからモジュールとして実行する。ルートの旧スクリプト名や
`python training/train_logkv.py`のようなファイルパスでの直接起動には対応しない。
既存の引数はそのまま使用できる。

```bash
# 通常のLogKV訓練とパイプライン訓練（GPU数・引数は環境に合わせる）
uv run torchrun --standalone --nproc_per_node=2 --module training.train_logkv --run-name myrun
uv run torchrun --standalone --nproc_per_node=2 --module training.train_logkv_pipeline --run-name pipeline

# 旧RecursiveCompressorのパイプライン訓練
uv run torchrun --nproc_per_node=6 --module training.train_pipeline

# 新旧アーキテクチャをconfig.jsonから判定する生成CLI
uv run python -m inference.predict_stream --model-dir /path/to/model
uv run python -m inference.predict --model-dir /path/to/model "日本の首都は"
uv run python -m inference.chat_server --help

# タスク別の実験（引数一覧）
uv run python -m exp.copying.train --help
uv run python -m exp.copying.evaluate --help
uv run python -m exp.selective_copying.train --help
uv run python -m exp.selective_copying.evaluate --help
uv run python -m exp.variable_memory.train --help
uv run python -m exp.variable_memory.evaluate --help
uv run python -m benchmarks.benchmark_logkv_predict --help

# 現行の全テスト。凍結した実験アーカイブは収集対象に含めない
uv run pytest
```

Selective Copyingは共通の訓練・評価関数へタスクモジュールを明示的に渡す。
以前の`sys.path`や`sys.modules["task"]`の差し替えは不要となり、両タスクを同一プロセスでimportできる。
**成果物の保存先は従来どおり`$DATA_DIR/exp/selective-copying/`**。
ソースのディレクトリ名だけが`selective_copying`に変わった。

## Pythonからの利用と対応表

```python
from models.logkv.configuration import LogKVConfig
from models.logkv.attention import LogKV, LogKVBlock
from models.logkv.modeling import LogKVLM
from models.logkv.pipeline import LogKVLMPipelineStage

from models.recursive_compressor.configuration import RecursiveCompressorConfig
from models.recursive_compressor.modeling import RecursiveCompressorLM
from data_pipeline.dataset import get_tokenizer
```

| 旧パス | 現行パス |
|---|---|
| `logkv.py` | `models/logkv/attention.py` |
| `configuration_logkv.py` | `models/logkv/configuration.py` |
| `logkv_lm.py` | `models/logkv/modeling.py` |
| `logkv_lm_pipeline.py` | `models/logkv/pipeline.py` |
| `recursive_compressor.py` | `models/recursive_compressor/attention.py` |
| `configuration_recursive_compressor.py` | `models/recursive_compressor/configuration.py` |
| `recursive_compressor_lm.py` | `models/recursive_compressor/modeling.py` |
| `recursive_compressor_lm_pipeline.py` | `models/recursive_compressor/pipeline.py` |
| `train_logkv.py` / `train_logkv_pipeline.py` / `train_pipeline.py` | `training/`内の同名ファイル |
| `predict.py` / `predict_logkv.py` / `predict_stream.py` / `chat_server.py` | `inference/`内の同名ファイル |
| `dataset.py` | `data_pipeline/dataset.py` |
| `benchmark_logkv_predict.py` | `benchmarks/benchmark_logkv_predict.py` |
| `test_logkv*.py` | `tests/logkv/`内の同名ファイル |
| `test_lm.py` | `tests/legacy/test_lm.py` |
| `exp/selective-copying/` | `exp/selective_copying/` |

## 過去の実験の再現

`doc/experiments/`のコード・記録・ハッシュは当時の内容を保持している。
アーカイブのスクリプトは現行パッケージ向けには移植していない。
各レポートの再現用commitを別worktreeに展開するか、その実験が記載する凍結ソースの
実行手順に従う。過去のCLI・ソースパス・監査ハッシュは、その時点の配置を表す。

現在のコードで新しい実験を行う場合は、上記のモジュール入口を使う。
過去の位置表現ブランチ専用オプションが現行モデルで使えるという意味ではない。
実験の索引は[LogKV実験記録一覧](logkv-experiments.md)を参照。

## 移行後の確認（2026-09-25）

- 既存329件のテストと、両タスクの同時import・明示的なタスク受け渡しの追加4件が通過。
- 訓練・生成・実験・ベンチマークの14入口で`--help`が成功。
- LogKV / RecursiveCompressor × Copying / Selective Copyingの4通りで、CPUの小規模な
  1-step訓練・保存・再ロード・評価が成功。共有評価関数の入力転送は指定された`device`を使用する。
- 2GPUのパイプライン統合テストが通過。loss/勾配、再開時の重み・optimizer・乱数の一致、
  層分割変更、保存・終了、保存モデルのストリーム生成を確認。
- 可変桁実験の既存初期化ハッシュが一致。移動後の推論ベンチマークの実行・ソースハッシュ記録も確認。

本学習・性能比較実験は実施していない。統合テスト成果物は
`/mnt/raid0/RecursiveCompressor/experiments/layout-smoke-20260925/`、
固定桁タスクの動作確認は同階層の`layout-task-smoke-20260925/`に保存した。
