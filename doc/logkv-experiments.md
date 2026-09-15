# LogKV実験記録一覧

2026-09-15、[標準LLM＋学習可能減衰の5,000-step訓練](logkv-lm-learnable-decay.md)を23:02 JSTに開始。
ユーザー指定で6GPU、事前実測・評価器確認通過、全体見積もり5.42時間。結果は未確定。

2026-09-15、標準CausalConv＋既存`--learnable-decay`による[固定10桁の比較](logkv-learnable-decay.md)を19:33 JSTに完了。
各50k steps・164セル、事前確認込み2.93時間、GPU解放済み。Copying bestはT131072まで全41点で256/256、
finalは同T75/256。Selective bestはT1024が10→190/256へ改善、16ヘッド中4ヘッドが増幅へ転じた。
Selective T131072は完全一致0、Copying16Mは未評価。標準の固定減衰は変更していない。

2026-09-14、CausalConvを標準としてmainへマージした（`654f311`）。
[可変桁M10/16/32/64の追加実験](logkv-variable-memory.md)は19:12 JSTに両50k学習・880セルの評価・監査を完了。
Copying M10/T131072は238/256完全一致。一方M32/64と未学習M128は両タスクで完全一致0。
実測4.67時間（事前確認込み）、GPU解放済み。

2026-09-14 13:55 JST、`logkv-causal-conv`ブランチの[位置埋め込みなし＋CausalConv](logkv-causal-conv.md)が完了。
2層・各Blockのattention前に幅4のconvを追加し、固定10桁の既存対照と比較した。
両タスク各50k steps、best/final T131072までの監査を通過。Copyingは全41点で各256/256完全一致
（best/finalは同一重み）。Selective T64は94/112例完全一致、最長距離は0/0例。
本実行2.85時間。続くユーザー依頼によるCopying T16777216の追加評価も8/8完全一致（固定10桁）。
追加評価はGPU0のみ約7.6分で完了し、GPU解放済み。mainへのモデル変更のマージは行っていない。

実験の成功・不成功を含めて記録を集約する。mainの標準実装と実験ブランチの実装は区別する。
各レポートに学習条件、評価条件、再現用commit、全評価値と成果物の場所を記載した。
モデル重みはDATA_DIRに保存し、リポジトリには設定・ログ・評価値・図・集計コードを収録する。

| 比較 | コード | 記録 |
|---|---|---|
| 過去のLM・記憶タスク検証 | 各節に記載 | [設計・実験記録](logkv.md) |
| 重複あり／重複なし参照 | `c9c51c9` / `3b0ce51` | [重複除去](logkv-refine-experiments.md) |
| phase2／retrieval相対logit補正／位置表現なし | `7261652` | [相対logit補正](logkv-relative-position-experiments.md) |
| Compressor固定／学習可能logit減衰 | `77356e6` | [圧縮減衰](logkv-compressor-experiments.md) |
| 子位置別直交変換／相対K/V／併用／レベル減衰除去 | `24b360c` (`logkv-position-study`) | [可変桁数比較](logkv-position-study.md) |

位置表現の実験APIは[別紙設計](logkv-position-design.md)を参照。
固定10桁の評価と可変桁数・prefix付きの評価は学習条件が異なり、数値を直接対照として扱わない。

可変桁数比較は2026-09-08に全12 runを完了した（6 GPU、各50,000 steps、best/final計5,280評価行）。
Copyingの完全一致は未達。Selectiveでは相対K/VによるM10の改善と、併用＋減衰なしによる
短いTでのM16/M32の改善があった。全数値、失敗例、checkpoint差、72件の合成圧縮診断をレポートに収録した。

## 固定10桁・RoPE・診断の追加記録

2026-09-13、位置表現調整を一旦終了し、mainの既存アーキテクチャへ戻る方針となった。
本一覧以下の文書・実行コード・結果をmainへ収録したが、モデル本体は取り込んでいない。

| 比較・診断 | 実装または実験記録commit | 記録 |
|---|---|---|
| combined-no-decay、固定M10 | `5b5f2f1`、境界確認 `06fd75b` | [固定10桁](logkv-fixed10-combined.md) |
| 再圧縮時の表現変化 | `2e9d3d9` | [圧縮診断](logkv-compression-diagnostic.md) |
| phase2＋局所読み出し／圧縮logit RoPE | 実装 `afaea2b`、結果 `b9c6719` | [phase2＋RoPE](logkv-phase2-rope.md) |
| phase2除去、読み出し／圧縮RoPE、T131072延長 | 実装 `afaea2b`、結果 `08669e1` / `71001ac` | [RoPEのみ](logkv-rope-only.md) |
| 自己スロット除去と桁誤り診断 | `96067e7` | [自己スロット](logkv-self-slot.md) |
| 読み出しhead・score・gate介入 | `9e9d1dc` | [読み出し介入](logkv-readout-intervention.md) |
| 読み出しRoPE角度倍率 | 実装 `c01061a`、結果 `6679d88` | [回転角比較](logkv-rope-scale.md) |
| 元トークン位置軸で基準を揃えたKey圧縮 | 実装 `8dcfea4`、Copying `a597467`、Selective `537e538` | [Copying](logkv-aligned-rope.md) / [Selective](logkv-aligned-selective.md) |

新方式はいずれも固定10桁Copyingの全域完全一致と長距離Selectiveの改善を同時に確立していない。
標準phase2の過去の16M・8/8の結果は、代替位置表現の成功を示すものではない。
各方式1学習seed、best/final、評価数、固定／可変記憶長の差など、各レポートの制約を維持する。

## 実験コードと再現

`doc/experiments/`の既存スクリプト・結果・ハッシュ記録は元の内容を保存している。
これらは過去実験の再現・監査用であり、mainから全キャンペーンを再起動するものではない。
多くのスクリプトは`/mnt/raid0/RecursiveCompressor/experiments/`以下の凍結ソース・チェックポイントを参照する。
別環境での再現には、各レポートのcommit、依存環境、チェックポイント、保存先の用意が必要。
記録されたパスがないときにmainのモデルへ置き換えて実行すると同じ実験にはならない。
PNGはGit LFSの対象。

`exp/position_study/`などの実験コードは[原コードのアーカイブ](experiments/position-code-archive-20260913/README.md)にも保存した。
mainの通常の訓練・テストには含めず、元commitの別worktreeで実行する。
元の実験ブランチ`logkv-aligned-rope`は変更していない。

## mainの位置埋め込みなし再評価（2026-09-13）

[固定10桁の再評価](logkv-no-position-main.md)：ユーザーの指定により、重複なしmainのまま
phase_emb=FalseのCopying / Selectiveを各50k、best/final T131072まで評価した。
モデル実装は`3b0ce51`と同一。2026-09-07のnoneに既存データがあることを確認した上での
再実行であり、2026-09-14 01:02 JSTに164セルの評価・監査まで完了（本実行2.72時間）。
Copying bestのT64は31/256完全一致、T131072は1/256。Selectiveは両Tで0/256。

[3層との比較](logkv-no-position-3layer.md)：追加依頼により、同条件で`num_layers=3`を
GPU2/3に追加（今回のみ計4GPU）。共通部分の未学習重みを2層と揃え、桁別誤答も比較した。
9月14日04:14 JSTに追加164セルまで完了（本実行4.57時間）、全GPU解放済み。
Copying bestのT64は237/256へ改善するが、T131072は0/256、桁精度も71.91%→53.75%に悪化。
Selectiveはbest/final全41点で桁精度が改善し、T131072 bestは21.72%→62.38%。
3層bestのT256で第3桁が第2桁を繰り返すが、同構造のfinalでは第3桁249/256正解。
この特定の衝突を原理的な識別不能性とは扱わない。1 seed、層数と容量増加は未分離。
