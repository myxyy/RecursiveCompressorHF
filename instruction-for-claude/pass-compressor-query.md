## やりたいこと
`RecursiveCompressor`の各再帰階層で処理される`compressor_query`を次段に引き継ぎたい  
現状各段が学習可能な`compressor_query`を持ち、これを用いてコンテキストの圧縮を行っているが、これを引数として前段の同一再帰階層の情報を利用する形にしたい  

## コード
`RecursiveCompressor.step()`の引数`x`をリストとして処理する  
`x = [q1, q2, q3, ...]`となるとき処理の流れは以下のようになる：  
1. `q1`に対してチャンク毎のself-attention+FFN
2. `q1`を圧縮クエリ`q2`とcross-attentionして圧縮コンテキスト`q2a`を得る
3. `x = [q2a, q3, ...]`として`step()`を再帰すると出力`[q2b, q3b, ...]`を得る
4. `q1`と`q2b`のcross-attentionにより伸長コンテキスト`q1b`を得る
5. `q1b`に対してチャンク毎のself-attention+FFN
6. `[q1b, q2b, q3b, ...]`を出力

現状`RecursiveCompressor`が持っている`compressor_query`は`RecursiveCompressorLM`に移動し、初段`RecursiveCompressor`に圧縮クエリとして渡す  
すなわち初段の入力`[q1, q2, q3, ...]`は`q1`がテキストの埋め込み、`q2, q3,...`が`compressor_query`をバッチサイズ、コンテキスト長、`chunk_size`、`compress_size`に合わせてexpandしたものとなる  
最終段の出力`[q1b, q2b, q3b, ...]`のうち`q1b`を処理したものが`RecursiveCompressorLM`の出力となる

## 実装結果

### 変更概要
- `compressor_query`を`RecursiveCompressor`から`RecursiveCompressorLM`に移動
- `RecursiveCompressor.step(xs, hidden)`がリスト入力を受け付けるように変更
- 再帰時にdeeper queriesを`compress_size`倍にexpandし、帰還時にmeanでcollapse

### アーキテクチャ上の変更
旧設計: 各レイヤーが独立した`compressor_query`を保持。レイヤー間でクエリは伝播しない。  
新設計: `compressor_query`は`RecursiveCompressorLM`が保持。各レイヤーを通過する際にクエリが進化し、前段の再帰階層情報を後段に引き継ぐ。

### predict != forward について
新設計ではクエリがレイヤー間で進化するため、部分チャンク（predict時）と完全チャンク（forward時）でクエリの進化度合いが異なる。  
これは設計意図通りの動作であり、旧設計の`predict == forward`等価性は新設計には適用されない。  
`predict`は自己回帰生成用として独立に正しく動作する（決定的、step==predict一致）。