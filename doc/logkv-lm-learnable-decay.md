# CausalConv＋学習可能レベル減衰：5,000ステップの言語モデル訓練

2026-09-15、固定10桁の[学習可能減衰実験](logkv-learnable-decay.md)に続き、
`train_logkv.py --learnable-decay --max-steps 5000`による新規事前学習とヘッド・生成傾向の解析を行う。
**準備完了。本学習の結果は未確定。** 実験ブランチは`adjust-attenuation`。
ユーザーは今回のLLM訓練に限り最大6GPUの使用を許可した。8時間以上の見込みなら事前確認する。
標準16層を使い、ユーザーの別run（32層）の再開や上書きは行わない。

## 条件

| 項目 | 設定 |
|---|---|
| 凍結ソース | `ce360e3`、モデル・標準CLIを変更しない |
| モデル | d1024 / H8 / ff3072 / 16層 / C4、327,392,384パラメータ |
| 構造 | 重複なし、位相埋め込みなし、CausalConv幅4、gate/self slotあり |
| レベル補正 | `−i β[layer,head]`、128係数、全てlog(4)初期値、符号制約なし |
| その他 | kv_norm / v_norm_only / level_amplify無効 |
| 学習 | 新規5,000 optimizer steps、seed0、context2048（入力・教師は2047位置） |
| GPU・バッチ | 6GPU DDP、各4例、蓄積1、実効24例/step、計120,000 packed rows |
| 入力トークン数 | 245,640,000（loss対象はPAD除外） |
| 最適化 | 元trainerのMuon＋AdamW、lr2e-4、warmup1000、clip1、βはAdamW・weight decay0 |
| 精度 | fp32 master weights＋bf16 autocast、評価も同じ（βはfp32） |
| データ | 既存pretrain cache：日本語/英語Wikipedia、cc100-ja、MiniPile |
| 保存 | 1000 stepsごと、1000〜5000の全checkpointを保持 |
| 観測 | 10 stepsごとに全β、TensorBoard loss/grad/lr、1000 stepsごとに日本語3プロンプト |

[元ソース記録](experiments/logkv-lm-learnable-decay-20260915/source_manifest.json)、
[キャッシュ記録](experiments/logkv-lm-learnable-decay-20260915/cache_manifest.json)。
6GPUでの過去のLLM実験と実効バッチ24は揃うが、アーキテクチャ・実装が異なる過去runとの
性能差を減衰学習だけの因果効果とは扱わない。今回、新たに固定減衰の対照モデルは訓練しない。

ラッパーは元の`train_logkv.main()`を呼び、係数記録と実験専用controlファイルを追加する。
周期的なサンプル生成はseed=`12345+step`の独立したRNG区間で行い、訓練のRNGへ影響させない。
データキャッシュはRAID上の既存配列へのリンクを使い、準備済みなのでprefaultは省く。
データ順・損失・モデル演算・optimizer更新式を変更しない。
訓練loss/EMAは元trainerどおりrank0のミニバッチ値で、6GPUの平均ではない。

## 事前実測と範囲

6GPU・実際のデータ・同じ設定で30 stepsを実行した。後半20 steps平均2.768秒/step、
約17,700 tok/s、最大allocated 18.83 GiB/GPU（rank0観測）。
訓練3.84時間、15%余裕＋評価1時間を含む見積もり5.42時間。
[事前実測](experiments/logkv-lm-learnable-decay-20260915/preflight.json)。

30-stepモデルで、未学習評価入力の抽出、符号付き係数の保存値照合、loss計算、生成、
attention観測器の有無による出力bit一致を確認した。
[評価器確認](experiments/logkv-lm-learnable-decay-20260915/evaluation_smoke/review.json)。
事前GPU実測開始から7.5時間を全体の停止上限とし、評価は最大1時間。
失敗時は停止し、自動再試行・追加訓練をしない。訓練完了後はGPU0だけで下記の評価を実行する。

## 訓練後の解析

1. **係数**：128ヘッドのβ履歴、増幅へ転じたヘッド、層ごとの傾向。
   1000〜5000の保存重みと記録値を照合する。
2. **未学習packed rowのloss**：6rankのDistributedSampler(seed0, epoch0)で消費する
   randperm先頭120,000件を除外し、4ソース各32例を固定抽出する。
   元の係数・推論時だけβ=log(4)・推論時だけβ=0の3条件で同じ入力を評価する。
   後2者は学習済みモデルへの介入であり、各設定で再学習した対照ではない。
   packed rowの重複は避けるが、文書単位の重複除去をした外部評価セットではない。
3. **実際のattention**：各ソース1例、末尾512クエリでjoint softmaxのレベル別／自己スロットへの
   確率質量を層・ヘッドごとに集計する。gateや出力射影の前の値。
   小さい標本による記述であり、ヘッドの因果的機能そのものとは扱わない。
4. **生成**：日本語3プロンプト×seed0–2×temperature0.7/1.0、各最大1024新規トークン（18件）。
   追加で同3プロンプト・seed0・temperature0.7を最大4096新規トークン（3件）。
   top_p=0.9、repetition_penalty=1、自然なEOSで停止する。
   全21件の出力・token IDを保存し、EOS率、長さ、末尾四分位の文字bigram異なり率、
   token 4gram反復率、同一token連続数を集計する。異なり率0.5未満は記述用指標で、品質判定の代替ではない。

評価入力は学習開始前に固定し、結果に応じて選び直さない。生成はfp32重み＋bf16 autocastで、
直前の記憶タスクに用いたbf16重み評価とは精度条件が異なる。

## 実行と保存先

通常CLIで同じ訓練ハイパーパラメータを指定する例（実験は下記ラッパー経由）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 .venv/bin/torchrun --standalone --nproc_per_node=6 train_logkv.py \
  --run-name d1024-h8-l16-conv4-learnable-decay-5000 \
  --dataset-type pretrain --context-length 2048 \
  --d-model 1024 --num-heads 8 --d-ff 3072 --num-layers 16 --chunk-size 4 \
  --conv-kernel-size 4 --gated-attention --self-slot --learnable-decay \
  --batch-size 4 --grad-accum 1 --lr 2e-4 --warmup 1000 --max-steps 5000 \
  --seed 0 --log-interval 10 --sample-interval 1000 \
  --checkpoint-interval 1000 --max-checkpoints 6 --no-prefault
```

実際のラッパー・監視コードは[実験フォルダ](experiments/logkv-lm-learnable-decay-20260915/README.md)。
元データ・凍結ソース・重み・訓練ログ・評価は
`/mnt/raid0/RecursiveCompressor/experiments/logkv-lm-learnable-decay-20260915/`。
checkpointはその下の`data/checkpoints_logkv/d1024-h8-l16-conv4-learnable-decay-5000/`。
標準のmain構成や他runは変更しない。
