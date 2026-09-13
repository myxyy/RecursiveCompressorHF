# phase2を維持した局所RoPE比較

## 状態・実験段階

2026-09-09、CopyingとSelective Copyingの全6 runの50,000 stepsとbest/final評価を完了した。
**Copyingは3方式・両checkpointともT=8,192までの全33評価点で完全一致。Selectiveは短距離の一部で改善したが、長距離の完全一致は未解決。**

実装commit `afaea2b1d8b0bbc6e91aba25e1b9a3594297875b`を別worktreeに凍結して使用した。
Copyingは9月8日22:01:02〜9月9日02:43:21 JST（約4時間42分）、
Selectiveは02:43:27〜07:22:42 JST（約4時間39分）。CPU集計は07:22:43に終了した。
合計約9時間22分で、最大2 GPUを使用し、両タスクは独立した学習・checkpoint・評価とした。
GPUはすべて解放済みで、追加実験の待機ジョブはない。

ユーザーの追加指示で2 GPU以内の連続実験が承認されたため、Copyingの保存結果を検証してからSelectiveへ進んだ。
全体の当初見積もりは12〜14時間。Copying実測から再推定したSelective段階は約6.13時間で、8時間未満だった。
各段階の8時間上限内で完了した。
16M評価への拡張とmainへの実験コードのマージは保留している。

実装後のCPUテストは265件成功。GPU事前測定では50k stepsの学習時間をphase2約2.54時間、retrieval-rope約4.29時間、
compressor-rope約2.81時間と推定。最大T=2028・batch64のbackwardは全条件で成功し、
ピークallocatedは14.81 / 15.00 / 15.14 GiB。
同じ200個のTで計測し、3条件の初期重みSHA256も完全一致した。
GPU0でphase2→compressor、GPU1でretrievalを実行し、事前測定では各段階約6〜7時間と見積もっていた。
この見積もりは短い事前測定によるもので、実際の所要時間は上記のとおり短かった。
標準評価のtoken budget=2^19でもT2048/8192・各256例が全条件で完走。
読み出しRoPEの評価ピークallocatedは17.95 GiB未満だった。
[Copyingの全結果・図表](experiments/logkv-phase2-rope-20260908/copying/README.md)を保存した。

## Copyingの結果

3方式×best/final×33評価点の**198セルすべてでtoken/string accuracyが100%**だった。
各セル256例で、T=4,096と8,192も全10桁が正解。各方式のbestとfinalは異なる重みである。
過去のcombined-no-decayで発生した最初の未学習圧縮段への切替による急落は、今回の評価点では生じなかった。
ただし評価点間のすべてのTや、16Mでの保持を確認したわけではない。

![Copying評価](experiments/logkv-phase2-rope-20260908/copying/comparison.png)

## Selective Copyingの結果

完全一致率（%）。各欄は **best / final**、各T・checkpointで256例。
bestは訓練区間の指標で選んだcheckpointであり、評価Tごとの最良値を選び直していない。

| T | phase2 | 読み出しRoPE | Compressor RoPE |
|---:|---:|---:|---:|
| 16 | 57.42 / 56.25 | 84.77 / 73.44 | 68.36 / 67.97 |
| 32 | 24.22 / 22.27 | 58.98 / 42.19 | 35.94 / 35.94 |
| 64 | 9.38 / 12.11 | 26.95 / 9.77 | 23.05 / 20.70 |
| 128 | 3.52 / 4.69 | 14.84 / 7.03 | 5.86 / 7.03 |
| 256 | 1.56 / 2.34 | 5.08 / 1.56 | 3.12 / 2.73 |
| 512 | 1.56 / 1.95 | 3.91 / 1.56 | 1.56 / 2.73 |
| 1024 | 1.56 / 0.39 | 1.17 / 0.00 | 2.73 / 1.17 |
| 2048 | 1.17 / 0.78 | 1.17 / 0.00 | 0.00 / 0.00 |
| 4096 | 0.39 / 1.17 | 1.95 / 0.00 | 0.39 / 0.00 |
| 8192 | 0.78 / 0.39 | 0.00 / 0.00 | 0.00 / 0.00 |

- T=32では読み出しRoPEがbest/finalとも改善（24.22/22.27% → 58.98/42.19%）。
- T=64では読み出しRoPEのbestが24/256→69/256に改善した一方、finalは31/256→25/256。checkpointへの依存が大きい。
- 同じT=64でCompressor RoPEはbest 59/256、final 53/256で、どちらも今回のphase2対照を上回った。
- T=1,024以降の完全一致例は少なく、T=8,192では両RoPE方式・両checkpointとも0/256。長距離の改善は確認できない。

T=8,192の桁正答率も、phase2 best/final 53.79/57.58%、読み出しRoPE 53.52/43.20%、
Compressor RoPE 53.83/51.37%であり、finalの長距離保持は対照より低い。

![Selective評価](experiments/logkv-phase2-rope-20260908/selective-copying/comparison.png)

## 学習時間と解釈

| タスク | 方式 | 学習時間（分） | best step |
|---|---|---:|---:|
| copying | phase2 | 137.78 | 37800 |
| copying | 読み出しRoPE | 231.66 | 31100 |
| copying | Compressor RoPE | 142.71 | 49800 |
| selective-copying | phase2 | 135.74 | 49000 |
| selective-copying | 読み出しRoPE | 227.78 | 48100 |
| selective-copying | Compressor RoPE | 141.65 | 49000 |

読み出しRoPEは今回の実行時間で対照の約1.68倍、Compressor RoPEは約1.04倍だった。
別GPUでの学習実時間の比較であり、厳密に統制したカーネル性能測定ではない。

今回の目的に対しては、局所RoPEをlogitに限定する方法が、少なくともT8192までのCopyingを維持しながら
Selectiveの短距離精度を改善し得ることを確認できた。ただし、長距離Selectiveや16M Copyingの解決には至っていない。
Compressor側はT64のbest/final両方で改善し、学習時間の増加も小さいため追加検証の候補になる。
読み出し側も短距離のbestは有望だが、finalの悪化と計算時間増加を踏まえる必要がある。

各条件1学習seedで、優位性の再現性は未評価。今回のphase2は新しく学習した対照で、過去のrefined phase2とは
実測精度が異なるため、方式間の比較には今回の対照を用いた。タスクごとのモデルは別であり、
同一重みによるCopyingとSelectiveの両立も未検証。追加の学習・長距離評価は本確認時には開始していない。

## 成果物の確認

全18コマンド（学習6、評価12）が正常終了。自動集計は各タスク198セル・合計396セルについて、
学習50k完了、best選択、条件差分、正解数とサンプル数の整合性を検証した。
9月9日の確認では保存ファイルのチェックサム、元データとのバイト一致、評価ログの全396点とJSONの一致を追加確認した。
モデル実装に変更はないため、既に成功した265件のテストは再実行していない。

[Copying全結果](experiments/logkv-phase2-rope-20260908/copying/README.md)、
[Selective全結果](experiments/logkv-phase2-rope-20260908/selective-copying/README.md)、
[確認記録](experiments/logkv-phase2-rope-20260908/review.json)、
[継続実行記録](experiments/logkv-phase2-rope-20260908/continuation.json)を参照。

## 設計

phase2（周期16）・固定レベル減衰・gated attention・self slotを維持する。
新方式は任意の有効化フラグで、既定値はどちらもfalse。パラメータやhidden形式を追加しない。

| モード | retrieval_rope | compressor_rope |
|---|---|---|
| phase2 | false | false |
| retrieval-rope | true | false |
| compressor-rope | false | true |

隣接する特徴次元をペアにし、周波数`θ_r = 10000^(−2r/d_head)`で回転する。
全head次元に適用し、周波数は全階層・headで共通、固定値。
回転と三角関数は最低fp32で計算し、logitの内積前に入力dtypeへ戻す。
有効化時のhead次元は偶数が必要。学習パラメータ・乱数消費は追加しない。

読み出し側は、階層iの現在のブロック内でQueryが属する子をj、完成済み子をcとして、
`q^T R(c−j) k / sqrt(d_head) − i log(C)`とする。
参照可能な距離j−cは1〜C−1で、絶対トークン距離や圧縮段数を角度に使わない。
self slotの相対距離は0、従来の内積を維持する。

Compressor側は`q_last^T R(c−(C−1)) k_c / sqrt(d_head)`でpooling重みを計算する。
その重みで**回転前のK/V**を混合し、Qは元のchunk-last queryを返す。
局所RoPEは保存するK/Vを直接回転しないが、pooling重みや下流層の入力は変わるため、
従来の記憶表現や長距離精度が不変になることを保証するものではない。

実装は相対角度でKeyだけを一時回転する。
実数の厳密演算ではQ/Kをそれぞれj/cで回転する式と等価である。
既存の相対K/Vも同時に指定した場合は、相対Key加算後に回転するが、今回の比較では指定しない。
子位置K/V変換、scalar相対bias、compressor decay、KV/V normも全条件で無効。

## 学習・評価条件

- 固定M=10、P=0。両タスクは元の専用データ生成器をそれぞれ使用
- d512、8 heads、d_ff1024、2層、C4、50,000 steps
- batch64、grad accumulation1、T~loguniform[1,2028]、全位置CE
- AdamW、lr3e−4、warmup1000、weight decay0、gradient clip1
- 学習seed0、データseed1。モード間で同じ初期重み・データ列を使用。各条件1 seed
- 学習はfp32重み＋bf16 autocast。best選択は従来の訓練100 steps区間のstring/token/−EMA loss
- 各best/finalをT=1〜14、16〜8192の2の冪と中間点、33点×256例で評価
- 評価seed12345、bf16重み＋autocast。結果は最後の10桁で採点、予測を入力へ戻さない

今回の段階では圧縮6段への外挿とその先のT8192までを調べた。
この範囲で成功しても16M保持を達成したとはみなさない。

## 検証

独立oracleは複素数でQ/Kを別々に回転し、productionの実数ペア・相対Key回転とは別経路で計算する。
C=2/3/4、読み出しのみ・圧縮のみ・併用について、fp64のforward値・入力とパラメータの勾配・
分割stepとpredict・因果性を検証した（値は絶対誤差1e−12以下）。
Compressor出力が回転前のK/Vの混合であること、config保存/読込、bf16有限出力、
無効時の旧config互換性とモード間の初期重み一致も確認した。

[事前測定スクリプト](experiments/logkv-phase2-rope-20260908/benchmark.py)、
[単一タスク・単一段階の実行スクリプト](experiments/logkv-phase2-rope-20260908/run_stage.py)。

Selective Copying段階も完了し、保存結果のCPU検証が成功した。
[Selectiveの全結果・図表](experiments/logkv-phase2-rope-20260908/selective-copying/README.md)を保存した。
