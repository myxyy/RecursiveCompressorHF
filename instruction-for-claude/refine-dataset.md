## やること
データセットに英語を混ぜる、あと形式を整理

## データセット
事前学習に会話データセットを入れていたのをやめて事前学習データセットと会話データセットを分ける

### pretrainデータセット
* `wikimedia/wikipedia`
  - `20231101.ja`のサブセット及び
  - `20231101.en`のサブセットを用いる
* `hotchpotch/cc100-ja-documents`
* `JeanKaddour/minipile`

### instructデータセット
* `shi3z/ja_conv_wikipedia_llama2pro8b_30k`
* `shi3z/ja_conv_wikipedia_orion14B_100K`
* `HuggingFaceH4/ultrachat_200k`

## 形式
ドキュメントデータセットのプリフィックスとしていた`[DOC]`タグは外し、bosトークン`<s>`だけ先頭に付いた形とする  
またコンテキスト長を超えるはみだし部分は単純に切り捨てるのではなく、コンテキスト長単位で分割してそれぞれを一つの文字列として扱った上で短い文章の結合等前処理を行う

例：
"abcdefghij", "123", "あいうえお"の文字列を1文字1トークンとしてコンテキスト長8で処理すると

1. まずbosトークンを先頭に付ける："<s>abcdefghij", "<s>123", "<s>あいうえお"
2. コンテキスト長単位で分割："<s>abcdefg", "hij", "<s>123", "<s>あいうえお"
3. コンテキスト長に収まるよう短い文字列を結合："<s>abcdefg", "hij<s>123", "<s>あいうえお"
4. コンテキスト長に足りない部分は<pad>トークンで埋められる："<s>abcdefg", "hij<s>123<pad>", "<s>あいうえお<pad><pad>"

という流れになる

前処理後の.mmapファイルはv3サフィックスとして`cc100_ja_v3.mmap`のようにしてください。

## その他
必要なテストがあれば適宜追加してください  
またこの文章も適宜追記修正お願いします

## 実装結果

### `dataset.py`
- `format_document` を削除（`[DOC]`プリフィックス廃止）
- `_units_doc_item` は生のtextを返すよう変更
- 旧 `_tokenize_unit` と `_pack_units` を廃止し、以下の2つで置き換え:
  - `_text_to_chunks(tokenizer, text, context_length)`: BOS付きトークン列をcontext_length単位で分割
  - `_pack_chunks(chunks, context_length, pad_token_id)`: チャンクを連結してパック（**末尾BOSは追加しない**、次チャンクの先頭BOSが区切り役）
- 長文の切り捨てを廃止 → context_length単位で分割して全データを学習に使用

### データセット構成
| dataset_type | 構成 |
|---|---|
| `pretrain` | wiki_ja, wiki_en, cc100_ja, minipile（文書のみ） |
| `instruct` | shi3z_llama2pro, shi3z_orion14b, ultrachat_200k（対話のみ） |

### キャッシュ命名
全データセットを`_v3`サフィックスで命名し、再ビルドする:
- `wiki_ja_v3.mmap`, `wiki_en_v3.mmap`, `cc100_ja_v3.mmap`, `minipile_v3.mmap`
- `shi3z_llama2pro_v3.mmap`, `shi3z_orion14b_v3.mmap`, `ultrachat_v3.mmap`

### テスト
- 旧 `test_pack_units_*` を `test_pack_chunks_*` に置換
- `test_text_to_chunks_short`, `test_text_to_chunks_long` を追加
- `test_pack_chunks_basic` で仕様書の例（"abcdefghij", "123", "あいうえお"）が正しく処理されることを確認
- `test_pack_chunks_no_trailing_bos` で末尾BOSが追加されないことを確認
- `test_units_doc_item_no_prefix` で[DOC]プリフィックスが付かないことを確認
