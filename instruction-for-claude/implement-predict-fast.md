# やること
`RecursiveCompressorLM.predict()`を用いて前回出力と隠れ状態を引き継いで1トークンずつ生成するコード`predict_fast.py`を作成する。

# モデル説明
`RecursiveCompressorLM.predict()`は単一トークンと隠れ状態を入力とし次トークン予測と次隠れ状態を出力とする。  
隠れ状態を引き継ぎながら1トークンずつ生成することで文章の生成を行う。  

# predict.py
コマンドライン引数は`predict_naive.py`（`predict.py`をリネームした）と同じにする。

# 実行方法
```bash
uv run python predict_fast.py "吾輩は猫である。" --context-length 256 --temperature 0.8
```

# 修正したバグ

## RecursiveCompressorLM.predict
- `hidden=None` 時に各レイヤーの隠れ状態が保存されない問題を修正。`hidden` を `[None] * len(self.layers)` で初期化するように変更。

## RecursiveCompressor.predict
1. **decompressor残差接続の不整合**: `inner_context_`（残差）がpost-FFNではなくpre-FFNの値を参照していた。`inner_context_ = inner_context` をFFNブロック後に移動。
2. **outer_contextの更新タイミング**: 圧縮結果を即座に`outer_context`として使っていたため、現在のチャンクの情報が漏洩していた。`next_outer_context`に分離し、decompression後に更新するように変更。
3. **norm_compressorの適用タイミング**: forwardでは`num_chunks > 1`の場合のみ`norm_compressor`を適用していたが、predictではチャンク満杯時のみ適用していた。forward/predictの両方で常に適用するように統一。

## RecursiveCompressor.forward
- `norm_compressor`を`num_chunks > 1`の条件から外し、常に適用するように変更。predictとの一貫性を確保。学習時はcontext_length >> chunk_sizeのため単一チャンクケースは事実上発生せず、影響は軽微。

# テスト
`test_predict_matches_forward` テストにより、forward/predictの出力一致を複数のシーケンス長（1, 7, 8, 16, 24, 32）で検証済み。