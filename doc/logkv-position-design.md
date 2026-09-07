# 位置表現実験の設計記録

この文書は実験ブランチの設計記録です。mainの標準実装とは区別してください。
retrieval相対logit補正は `7261652`、Compressor logit減衰は `77356e6`、
子位置別K/V変換と相対Key/Valueは `24b360c` に実装されています。
再現時は該当commitを別worktreeにcheckoutして実行してください。
子位置別変換・相対K/Vの詳細は[可変桁数比較](logkv-position-study.md)を参照。

### 2.4 サブチャンク間の相対位置logitバイアス

`relative_position_bias=True` で、全レベルの有効スロットに次の補正を追加する。
コードのレベル番号は最下位がi=0（提案図の「レベル1」）。

```text
j = floor(s / C^i) mod C
distance = j - c - 1                  # 有効スロット c < j では0以上
logit = q·k / sqrt(d_head) - i*log(C) - distance*log(C)/(C-1)
```

C=4なら追加係数はlog(4)/3で、レベルi=1では同じ4トークンのサブチャンク内で一定。
直前の完成済みサブチャンクへの追加減衰は0、それより古いスロットは距離に応じて減衰する。
内容logitが同じとき、非正規化重みは `C^(-i-distance/(C-1))` に比例する。
既存のcausal mask・重複なし参照・全レベル共通softmaxを維持する。
self slotのバイアスは0。このオプション単独では圧縮用attention poolingは変更しない。
`learnable_decay` / `level_amplify` を使う場合も追加の相対項は固定のまま。

追加の学習パラメータやhiddenは不要。C=2では各レベルの有効スロットが最大1個なので、
追加距離は常に0となる。位相埋め込みと同等の表現力を保証するものではなく、特に
圧縮済みチャンク内部の順序保持への効果は実験で確認する。

互換性のためConfigのデフォルトはFalse。既存チェックポイントの挙動は変わらない。
位相埋め込みを置き換える実験では `--relative-position-bias` を指定し、
`--phase-emb` は付けない（`phase_emb=False`）。既存の固定レベル減衰は引き続き有効。
Copying / Selective CopyingとLMの両学習CLIで同じオプションを利用できる。

### 2.5 Compressorの位置減衰

`compressor_decay=α` を指定すると、圧縮のsoftmax直前に次を適用する。
圧縮は末尾Queryを用いるため、古い順の距離は `[C-1, ..., 1, 0]`。

```text
compression_logit[c] = q_last·k[c] / sqrt(d_head) - α*(C-1-c)
```

KとVはこの同じ重みで圧縮し、q_outは従来どおり末尾q。既存のretrieval側の相対項や
固定レベル減衰とは独立した設定。レベル内で共通の `i*log(C)` を圧縮softmaxに加えても
相殺されるため、圧縮側には距離項だけを入れる。

- `--compressor-decay 0.46209812037329684`: C=4の基準 `log(C)/(C-1)`。
  内容logitが等しい場合、最古／最新の重み比は1/4。半分の係数なら1/2
- `--learnable-compressor-decay` も指定すると `α_h=softplus(raw_decay_h)`。
  正の `compressor_decay` を初期値の最大値とし、Hヘッドを最大値〜その1/4の等比列で初期化
  （H=1は最大値）。各Transformer層・ヘッドで別パラメータを持ち、全圧縮階層で共有。
  d512/8 heads/2層なら追加16パラメータ。初期化は乱数を消費しない
- デフォルトは `compressor_decay=0.0, learnable_compressor_decay=False`。
  この場合は追加のパラメータがなく、従来の出力・勾配と一致。負・非有限な係数と、
  学習可能版の初期値0は拒否する
- 複数チャンクのバッチは `(batch, head, chunk)` の順で平坦化される。
  `Compressor.forward(..., num_chunks=N)` にNを渡し、ヘッドと係数を対応させる。
  hiddenの形式と対数サイズは変わらない。既存の実行中hiddenを新しい設定へ流用せず、
  設定変更後は `hidden=None` から始める
- 有効時はlogit補正・softmaxを最低fp32で行い、K/Vとの積の前に重みのdtypeを合わせる。
  fp64は維持し、bf16重みのautocastなし推論にも対応する

この補正はK/Vの順序による非対称性を圧縮に導入するが、同一valueの正規化加重平均が
同一になる性質は残る。固定係数が大きいと再帰的な圧縮で古い情報への減衰が重なるため、
係数と記憶精度の関係は実験で確認する。
位相埋め込みを使わない実験では `--phase-emb` を付けず、retrievalの相対補正を併用するなら
`--relative-position-bias` を指定する。

検証: 221 tests passed。独立した圧縮・retrieval参照とのfp64出力・勾配一致、C=2/3/4、
ヘッドとチャンクの対応、任意分割・逐次推論、因果性、再計算、bf16、保存読込を含む。
