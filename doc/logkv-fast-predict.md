# 1トークン推論専用のpredict

2026-09-16、`logkv-fast-predict`ブランチで`LogKV.predict()`の`step()`依存を除去した。
訓練済み重み、モデル設定、固定レベル減衰、参照区間の意味論を維持する最適化。
訓練は実行していない。

## 実装

時刻sの入力hiddenでは、各レベルの未完成チャンクに`floor(s/C**i) % C`個の要素がある。
これらは過去`[0,s)`を重複なく分割した、まさに今回のQueryが参照するスロットである。
この性質により、1トークンの場合は次の順序で処理できる。

1. Q/K/Vを射影し、更新前のキャッシュからattentionを計算する。非空レベルをまとめて
   小さな行列演算に渡すため、汎用stepの絶対位置計算・gather・レベルごとのsoftmax処理を省ける。
2. 今回のQ/K/Vを最下位に追加する。C個揃った場合だけ圧縮して上位へ渡す。
   上位も揃えば繰り返す。C進数の繰り上がりに相当し、通常のトークンでは最下位だけを更新する。
3. gateと出力射影を適用する。

Self slotは今回の未圧縮K/Vを追加する。位相埋め込み、固定減衰・増幅、学習可能減衰、
K/V norm、gateの既存オプションに対応する。
状態は従来の`(levels, offset)`で、入力hiddenを破壊せず、推論時の保存量は引き続き
`O(C log L d)`。完成したチャンクの大きなストレージを参照として残さない。

`CausalConvBlock.predict()`は1窓の畳み込みと履歴更新を行う。
`LogKVBlock.predict()`、`LogKVLM.predict()`も下位の専用predictを呼ぶ。
HF `generate()`はprefillをstepで処理し、その後のキャッシュ付き1トークンforwardを
勾配無効時にpredictへ振り分ける。生成CLIに追加引数は必要ない。
勾配有効のforwardと複数トークン処理は既存のstepを使用する。

```python
model.eval()
with torch.inference_mode():
    logits, hidden = model.step(prompt_ids)  # (B, prompt_length)
    token = logits[:, -1].argmax(-1)
    logits, hidden = model.predict(token, hidden)  # token: (B,), logits: (B, vocab)
```

## 数値精度と検証

全レベルを単純に1個のbf16 softmaxに置き換えると、既存のレベル内value積の丸め位置が変わる。
その差を抑えるため、レベル内の最大logit・分母・value積を保持してから、fp32以上で統合する。
固定biasと学習可能biasの既存のdtype規則も維持する。
ただし演算のまとめ方・カーネル・レベル間加算順が異なり、bf16のbit一致は保証しない。
小さなlogit差でもargmaxやsamplingの境界を跨げば生成列は変わり得る。

以下の212テストが通過した（GPUテストはRTX 3090の1台のみ使用）。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest \
  test_logkv.py test_logkv_lm.py test_logkv_conv.py test_logkv_predict.py -q
```

- 独立した`reference_forward`とのfp64一致（既存の`<1e-12`検査を含む）、C=2/3/4、H=1/4、各種オプション。
- Cの累乗境界を跨ぐ圧縮、全トークンpredict、step→predict→step、状態の非破壊・保持ストレージ量。
- Convを含む勾配の一致、CPU/CUDAでのfp32・bf16重み・bf16 autocast、負や0の学習可能減衰。
- LM/Block/attention/Convのstepを呼べなくしてもpredictが動作すること、HF decodeの専用経路利用。

## 速度比較の条件

ベンチマークは[benchmark_logkv_predict.py](../benchmark_logkv_predict.py)。
RTX 3090 1台、batch=1、C=4、d1024/H8、gate/self slot有効。
LMは直近の固定減衰5000-step重み（16層、ff3072、Conv幅4）をそのままロードする。
通常の訓練や生成品質評価ではなく、同じprefill状態から同じランダム入力列128トークンを処理する
teacher-forcingによるdecode速度比較である。

stepとpredictを交互順で各3回測定した中央値。前後でCUDA同期し、Python起動やkernel発行も含む
wall-clock時間を測る。prefill、重みロード、warmup、精度確認、sampling/tokenizerは計時から除外する。
attention単体の数値をLM全体や長文prefillの高速化率として扱わない。

結果JSONは[experiments/logkv-fast-predict-20260916/](experiments/logkv-fast-predict-20260916/)に保存する。
各JSONに全反復の時間、誤差、環境、ソースSHA256、LM重みSHA256を収録する。

## 測定結果

| 対象 | 精度 | prefill長 | 従来step（ms/token） | predict（ms/token） | 高速化 |
|---|---|---:|---:|---:|---:|
| attention単体 | bf16重み | 2,048 | 4.210 | 1.042 | 4.04倍 |
| attention単体 | bf16重み | 16,384 | 5.303 | 1.033 | 5.14倍 |
| 学習済み16層LM | fp32重み + bf16 autocast | 2,048 | 75.287 | 25.088 | 3.00倍 |
| 学習済み16層LM | bf16重み、autocastなし | 2,048 | 72.222 | 22.322 | 3.24倍 |

LM全体では約13.3→39.9 token/s（autocast）、約13.8→44.8 token/s（bf16重み）。
これらは128トークンの特定区間での測定で、バッチサイズ、状態の埋まり方、モデルサイズにより変わる。
GPUは測定終了後に解放した。

同じ入力128トークンについて、それぞれ独立に状態を更新した結果の差は次の通り。

| 対象 | 出力の最大絶対差 | RMSE | 最大logitのトークン一致 |
|---|---:|---:|---:|
| attention、prefill 2,048 | 0.000977 | 0.00000436 | 対象外 |
| attention、prefill 16,384 | 0.000977 | 0.0000102 | 対象外 |
| LM、autocast | 0.0625 | 0.006779 | 128 / 128 |
| LM、bf16重み | 0.093262 | 0.012897 | 127 / 128 |

**bf16重みのLMでは1箇所でargmaxが変わった。** したがって既存生成結果のbit再現や、
自己回帰生成列の完全一致を保証する最適化ではない。上表の一致数はランダム入力を固定した
teacher-forcingであり、自由生成の一致率・生成品質・Copyingの再評価を表すものではない。
従来の1トークン演算を使う場合は`model.step(token[:, None], hidden)`を利用できる。

重みは[固定減衰LM比較](logkv-lm-fixed-decay.md)のcheckpoint-5000。
`model.safetensors`のSHA256は
`ed1c8e2e5800a40ca67842870472e6f92144fa8d1c4372b4c39436fc5a1f2095`。

## 再現

再現例：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python benchmark_logkv_predict.py \
  --target attention --mode bfloat16 --prefill 2048 --tokens 128 --repeats 3

# 実際に比較するローカルの学習済みモデルを指定する。
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python benchmark_logkv_predict.py \
  --target lm --checkpoint /path/to/checkpoint-5000/model \
  --mode autocast --prefill 2048 --tokens 128 --repeats 3
```
