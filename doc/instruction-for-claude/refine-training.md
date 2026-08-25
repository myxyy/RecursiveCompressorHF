# 訓練の改善
## やること
現状（459d2586d17c8f81693d5b548af322bbaf1d04d7）の設定でしばらく学習したがlossが3.5あたりから下がらなくなっている。試しに出力させてみてもあまり芳しい結果が得られておらず、モデルサイズとデータセットの両面から改善を試みる

## 現状の出力例（文章生成）
myxy@myxy-ML3:~/RecursiveCompressor$ uv run predict.py "[DOC]吾輩は猫である" --model-dir /mnt/raid0/RecursiveCompressor/checkpoints/checkpoint-42367/ --context-length 256 --temperature 0.8
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 436/436 [00:00<00:00, 4156.43it/s]
<s> [DOC]吾輩は猫である Canada

Places


External links 
 Caledonian foldetsblen
19539
List of mammals and collectites

Erect of the Royal Irish Empire
Hamphooe letters records by the name of the British Waelic



References
Citations

 23
Colett)<s> [DOC]Oxheanai
Ginamon languages<s>us
Clavic cider genera
Cimicidae<s> [DOCCritical uses
South governors<s>
Moths in 
Bacteria described in 153<s> [DOCTaxa
corporated communities of Belgon<s> [DOC
Gaspnogenian<s><s> [DOC

Spineopoda<s> district<s> [DOC<s>


Tachrigicaceae genera<s><s><s><s>
Plants described by its names<s>
Further reading<s> [DOC 




Bacteria
 
 Gloria gases<s> [DOCCambialaide<s> [DOC]Carronotbra (demolcifolia)

myxy@myxy-ML3:~/RecursiveCompressor$ uv run predict.py "[QUERY]日本の首都はどこですか[ANSWER]" --model-dir /mnt/raid0/RecursiveCompressor/checkpoints/checkpoint-42367/ --context-length 256 --tempera
ture 0.8
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 436/436 [00:00<00:00, 4146.76it/s]
<s> [QUERY]日本の首都はどこですか[ANSWER]はい、Alias Island () is a village in the municipality of Portofie Chihan, Mohanoi Township,  isgar, in the district of the district of Congo.


References


External links
Towns in Kuruk District<s> [DOC]Townq-e District<s> [DOC]
Town<s> [DOC
Rural localities in Etigraphodet (zhah Province)
Towns, Iran<s> [DOC]sky District<s> [DOC]水中城
 City<s> [DOC]ポルト (yehin Province (; district) district)
References

Rural localities in Aleiksky District<s> [DOC]
Rural localities in Soko District<s> Region
Rural locality<s>rovsky District<s> [DOC]Nakhshayaayevsky District<s> [DOC]/* information on the following rural rural locality of the population of the west of the Central district was 53060 people
Towns.
North-ye Shoghuovo District<s> [DOC]I haltarya settlement in Tiyengz

## 課題点
* 出力が英語に偏っている
  - 日本語の入力に対しても英語出力になってしまう
* 質問に対しての回答が文章生成になってしまう
  - ファインチューニングで解決可能か
* モデルサイズが小さい（0.3B）

## 改善案
### 日本語データセットのみにする
多言語学習によるメリットは大規模学習でしか得られないと割り切り、現状の計算資源では日本語のみに特化させる  
具体的にはデータセットを以下の3つに制限  
* `wikimedia/wikipedia`
  - `20231101.ja`のサブセット
* `shi3z/ja_conv_wikipedia_llama2pro8b_30k`
* `shi3z/ja_conv_wikipedia_orion14B_100K`

### モデルサイズの拡大
夏目漱石データセットで試したときはlossが0コンマ台まで下がったがデータセットが大きくなると現状のモデルサイズ0.3Bでは足らなそう  
学習コンテキスト長を半分の2048、dtypeをbfloat16にしてモデルサイズを4倍にする  
RecursiveCompressorの積層数を4倍にするのがよさそう

## Claudeへの質問
質問：上記出力例だと`出力1<s>[DOC]出力2<s>[DOC]出力3<s>[DOC]...`のような形式の出力が見られますがもしかしてデータセットが
* `いろはにほへと`
* `ちりぬるを`
* `わかよたれそ`
の場合、私が意図していた
* `<s>[DOC]いろはにほへと[PAD][PAD][PAD]...`
* `<s>[DOC]ちりぬるを[PAD][PAD][PAD][PAD][PAD]...`  
* `<s>[DOC]わかよたれそ[PAD][PAD][PAD][PAD]...`  
のように1データずつパディングするのではなく
`<s>[DOC]いろはにほへと<s>[DOC]ちりぬるを<s>[DOC]わかよたれそ...`  
みたいにまとめちゃってますか？参考としてお見せしたレポジトリが確かそんな感じの実装だったので…単に末尾の`<s>`のあとに`[DOC]`を予測しているだけかもですが念のため確認です
回答：結合はしていません。`dataset.py`の`_tokenize_to_seq`で1サンプルずつ`<s>[DOC]テキスト<s>`の形にトークナイズし、残りをPADで埋めています。`MemmapDataset.__getitem__`でPAD位置のlabelsは-100にマスクされるので、PAD部分からは勾配が流れません。  
出力例の`出力1<s>[DOC]出力2<s>[DOC]...`パターンは、モデルが末尾の`<s>`の次に`[DOC]`が来るパターンを学習した結果、生成時にそれを繰り返しているだけです。

## Claudeからの追加改善案

### bfloat16対応
モデルサイズ拡大に伴いVRAM消費を抑えるため、bfloat16での学習は有効。`RecursiveCompressorConfig`に`torch_dtype`を設定し、モデル初期化時に`model.to(dtype=torch.bfloat16)`とする。

### gradient accumulation
バッチサイズを大きくする代わりに、gradient accumulationで実効バッチサイズを増やすことも検討可能。現状バッチサイズ1/GPUだが、accumulation_steps=4にすれば実効バッチサイズ24（6GPU×4）になり、学習が安定する可能性がある。

### PAD割合の問題
context_length=4096に対してWikipediaの1記事が短い場合、大半がPADになり計算効率が悪い。対策として:
- 短い文書を複数結合してcontext_lengthを埋める（ただし現状の要件と矛盾するため要検討）
- context_lengthを文書長の分布に合わせて調整

### ログにloss以外の指標を追加
学習率、勾配ノルム、スループット（tokens/sec）等を表示すると学習状況の把握に役立つ。