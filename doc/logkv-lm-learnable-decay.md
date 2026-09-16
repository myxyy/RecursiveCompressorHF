# CausalConv＋学習可能レベル減衰：5,000ステップの言語モデル訓練

2026-09-15、固定10桁の[学習可能減衰実験](logkv-learnable-decay.md)に続き、
`train_logkv.py --learnable-decay --max-steps 5000`による新規事前学習とヘッド・生成傾向の解析を行う。
**2026-09-16 03:33:21 JSTに5,000-step学習・生成21件を含む評価・自動監査が完了した。**
本実行4.51時間、事前GPU確認込み4.61時間。GPU解放済み。追加訓練は開始していない。
同日の完了後解析ではPADを除いたattention再計測をGPU0で24.7秒実施し、終了・解放した。
5個のcheckpoint、501回の係数記録、全生成のtoken/文字・反復指標を再監査した。
[完了後監査](experiments/logkv-lm-learnable-decay-20260915/analysis/review.json)。
本学習は2026-09-15 23:02:55 JSTに開始。
準備commit `5207743`。当初の完了目安は9月16日4:30 JST頃、停止上限は同6:26:50 JSTで、期限内に完了。
[開始・監視記録](experiments/logkv-lm-learnable-decay-20260915/campaign.json)。 実験ブランチは`adjust-attenuation`。
ユーザーは今回のLLM訓練に限り最大6GPUの使用を許可した。8時間以上の見込みなら事前確認する。
標準16層を使い、ユーザーの別run（32層）の再開や上書きは行わない。

後続の[固定減衰での再学習比較](logkv-lm-fixed-decay.md)も同日完了した。
同条件の固定版lossは3.62275、学習可能版3.61796。低温反復は両方4/9で、固定版でも解消しなかった。
以下は最初の学習可能版解析を保存した記録であり、対照実施後の判断は後続レポートを参照。

## 結果と考察（2026-09-16）

**今回のLMは全ヘッドで減衰を維持しながら弱める方向を学習し、その係数を使うと評価lossが最小だった。
ただし低温生成の反復は残り、固定減衰で再学習したモデルに対する優位性は未検証。**

### 学習と係数の推移

5,000 stepのrank0最終lossは3.3732、EMAは3.5218。全stepのloss・grad norm・lrは有限で、
記憶タスクの一部で見られたような終盤の大きな悪化は訓練EMAには現れていない。
ただし固定した評価集合のlossを途中checkpointごとに測ったわけではない。

| step | rank0訓練EMA | β最小 | β平均 | β最大 |
|---:|---:|---:|---:|---:|
| 0 | — | 1.3863 | 1.3863 | 1.3863 |
| 1,000 | 4.6088 | 1.2447 | 1.2844 | 1.3421 |
| 2,000 | 3.9874 | 1.0480 | 1.1833 | 1.2980 |
| 3,000 | 3.8068 | 0.9042 | 1.1166 | 1.2807 |
| 4,000 | 3.6286 | 0.7850 | 1.0643 | 1.2996 |
| 5,000 | 3.5218 | 0.6850 | 1.0215 | 1.3295 |

最終128/128ヘッドが `0 < β < log(4)`。増幅するヘッドは0で、
最弱減衰は第9層H5（0.6850）、最強は第7層H4（1.3295）。層・ヘッド番号は1始まり。
最後の1000 stepsでも121/128ヘッドでβが低下し、平均はさらに0.04284下がっている。
**まだ係数が収束したとは言えず、LMが将来も増幅を必要としないという結論ではない。**
Selectiveは50k、今回のLMは5kで、optimizer・学習率・データも異なる。

平均βは初期値の約74%。logitの差としては小さく見えても、レベルiで累積する。
例えばβ=1.0215のヘッドを仮定すると、i=5のバイアスだけのsoftmax分子倍率は
固定log(4)の場合の約6.2倍になる（q・kなど他項を固定した計算）。
このため「負に転じない＝圧縮記憶への扱いがほぼ不変」ではない。

![係数と訓練lossの推移](experiments/logkv-lm-learnable-decay-20260915/results/training.png)

[全ヘッドの最終係数図](experiments/logkv-lm-learnable-decay-20260915/results/beta_final.png)、
[全係数](experiments/logkv-lm-learnable-decay-20260915/results/coefficients.json)、
[集計](experiments/logkv-lm-learnable-decay-20260915/analysis/coefficients_summary.json)。

### 同じ学習済みモデルへの係数介入

未消費packed row 128例、PADを除く212,943教師トークンを評価した。
全条件でモデルの他の重みと入力は同一。下表の固定／ゼロは**推論時だけβを書き換えたもの**。

| 推論時のβ | 全体loss | Perplexity | 学習βからのloss増分 |
|---|---:|---:|---:|
| 学習値 | **3.61796** | **37.26** | — |
| 全ヘッドlog(4)に戻す | 3.63179 | 37.78 | +0.01383 |
| 全ヘッド0にする | 3.79552 | 44.50 | +0.17756 |

| 評価ソース | 学習β | β=log(4) | β=0 |
|---|---:|---:|---:|
| Wikipedia ja | 3.25097 | 3.27052 | 3.43861 |
| Wikipedia en | 3.40195 | 3.41080 | 3.60290 |
| cc100-ja | 3.46778 | 3.48100 | 3.67803 |
| MiniPile | 4.33264 | 4.34637 | 4.44521 |

4ソースすべてで学習値が最良。今回のcheckpointは弱めた減衰に適応しており、
係数を戻すとlossが上がる。ただしlog(4)との差は小さく、各例のloss分散は保存していないため
統計的有意性は判定していない。ゼロ初期化や固定減衰で最初から訓練した場合の成績は未測定。

[ソース別loss](experiments/logkv-lm-learnable-decay-20260915/results/heldout_loss.json)、
[トークン数で重み付けした全体値](experiments/logkv-lm-learnable-decay-20260915/analysis/loss_summary.json)。

### 実際のattention：局所参照と圧縮記憶参照の分化

**解析修正**：元の`results/attention_mass.json`は4例の絶対末尾512位置を平均しており、
PAD位置を含んでいた。日本語Wikipediaの例では有効な教師位置が0/512、英語Wikipedia144/512、
cc100-ja334/512、MiniPile512/512だった。この旧集計はヘッド解釈には使用しない。
学習やloss（PAD除外済み）、生成には影響しない。元ファイルは監査用に保存した。

同じ未消費128例に対し、各例の「入力が非PADかつ教師が非masked」の最後の最大512位置に
限定して再計測した。計64,338 query位置×128ヘッド。観測の有無による出力bit一致も4ソースで確認。
各queryを等しく重み付けし、下表の層の値は8ヘッド平均。
ここでレベルはコードのiで、L0が元token、L1が4token、L2が16tokenの圧縮単位。

| Transformer層 | 自己スロット | L0 | L1 | L2以上 |
|---:|---:|---:|---:|---:|
| 1 | 92.83% | 5.00% | 1.49% | 0.68% |
| 2 | 47.15% | 27.71% | 13.80% | 11.35% |
| 5 | 18.85% | 23.43% | 19.53% | 38.18% |
| 7 | 16.89% | 21.40% | 20.12% | 41.59% |
| 10 | 16.76% | 23.13% | 20.55% | 39.56% |
| 16 | 29.06% | 29.50% | 19.43% | 22.01% |

第1層は自己スロット中心、中間層で圧縮記憶への割合が増し、最終層では局所側に戻る。
ただし自己スロットの表現にも直前の幅4 CausalConvが入るため、自己スロットを選ぶことは
「過去を一切使わない」ことを意味しない。また、この割合は出力gate・射影前のsoftmax質量で、
最終logitへの因果的寄与の大きさそのものではない。

L2以上へ強く向けるヘッドは第7層H1が73.21%（β0.8513）、第9層H5が72.59%（β0.6850）、
第5層H7が67.93%（β1.1336）。**βが正でも内容のq·k項と合わせて圧縮記憶を強く参照できる。**
「増幅ヘッドだけが長距離参照担当」という解釈は狭すぎる。
この測定は最大2047入力位置の自然文上の観測であり、無限長外挿を検証したものではない。

![PADを除外したattention](experiments/logkv-lm-learnable-decay-20260915/analysis/attention.png)

[修正後の生値・PAD監査](experiments/logkv-lm-learnable-decay-20260915/analysis/attention_valid.json)、
[ヘッド別集計](experiments/logkv-lm-learnable-decay-20260915/analysis/attention_summary.json)、
[再計測コード](experiments/logkv-lm-learnable-decay-20260915/attention_valid.py)。

### 生成：低温の反復は未解決

| 最大新規token | temperature | 件数 | EOS | 平均長 | 末尾1/4文字bigram異なり率 | 同率<0.5 |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 0.7 | 9 | 4/9 | 674.2 | 0.461 | 4/9 |
| 1024 | 1.0 | 9 | 6/9 | 489.1 | 0.852 | 0/9 |
| 4096 | 0.7 | 3 | 3/3 | 1487.7 | 0.547 | 1/3 |

temperature0.7では「家を売る」から地名列へ移行して「高知県」を繰り返す例、
「宇宙飛行士」から「機動戦士ガンダム」の一覧反復へ移行する例、「廃油」の反復へ陥る例がある。
一方、同一tokenの最大連続数は全21件で3にとどまる。今回の主な問題は単一tokenの連打よりも、
複数tokenからなる語句・話題・一覧の反復。指標0.5以上でも「宇宙」への執着が続く例があり、
この閾値だけで良好な出力を判定できない。

temperature1.0ではこの強い反復指標を下回る例はなく、長く続いた出力でも表現が多様になる。
ただし日本語らしい文章であることと、事実性・話題整合性は別問題で、国・地名・年号などの
混同や話題逸脱は残る。「日本の首都は」に対して適切な短い回答を安定して返す段階ではない。
文書事前学習でありinstruction tuningは行っていない。

**4096上限の3例は独立の追加seedではない**。同じseed0の1024上限生成のprefixとtoken列が一致した。
「日本の首都は」は377token、「人工知能とは」は18tokenで先に終わり、
実際に訓練contextを超えて続いたのは「昔々あるところに」の1例だけ。
この例は4068tokenでEOSに達したが、末尾bigram異なり率0.060の地名反復が長く続いた。
したがって**EOS3/3を長距離生成の成功とは扱わない**。

[全生成](experiments/logkv-lm-learnable-decay-20260915/results/generations.json)、
[生成集計](experiments/logkv-lm-learnable-decay-20260915/results/generation_summary.json)、
[四分位推移・prefix照合](experiments/logkv-lm-learnable-decay-20260915/analysis/generation_analysis.json)。

### 現時点の判断と次の比較候補

学習可能減衰はLMでも働き、減衰を弱めた値への適応とヘッドの参照先の分化が確認できた。
一方、生成反復は重複のない構造でも残る。これをβの学習のせいと断定するには対照が不足している。
一律のゼロ化・増幅を選ぶ証拠は今回なく、標準設定は変更していない。

次に比較するなら、同じ現在の構造・初期化・データ順・実効バッチ・5k stepsでの
**固定減衰の再学習対照**が最も直接的。両モデルの同じ未消費例のlossと同じ生成設定を比較すれば、
学習可能化が反復を増やすのかを判断しやすい。より小さい診断としては、同checkpointのβだけを
log(4)へ戻した生成比較があるが、今回はその介入ではlossのみ測った。
継続学習での係数収束や生成の変化も候補になる。これらは提案で、追加実験は開始していない。

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

## 当初の評価計画（実施済み、attention修正は上記）

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
