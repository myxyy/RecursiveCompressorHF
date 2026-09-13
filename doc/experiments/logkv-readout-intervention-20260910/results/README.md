# 読み出し介入実験：CPU検証済み結果

既存checkpointでの単一回答位置への反実仮想介入。改善しても、そのまま実装可能な修正の評価ではない。
discoveryは既存seed4321の16例、validationは独立seed20260911の32例。
下表はT49152の9桁目の正答数。best/finalとも自己スロットあり・phase2なし。

| 介入 | best bf16 発見/検証 | final bf16 発見/検証 | best fp32 発見/検証 |
|---|---:|---:|---:|
| none | 3 / 16 | 16 / 32 | 5 / 18 |
| head8 | 16 / 32 | 16 / 32 | 16 / 32 |
| head1 | 3 / 16 | 16 / 32 | 5 / 18 |
| phase | 16 / 32 | 16 / 32 | 16 / 32 |
| mask | 3 / 16 | 16 / 32 | 5 / 18 |
| phase_mask | 16 / 32 | 16 / 32 | 16 / 32 |
| l1_block | 16 / 32 | 16 / 32 | 16 / 32 |
| l2_gate | 16 / 32 | 16 / 32 | 16 / 32 |
| l2_raw | 3 / 15 | 16 / 32 | 5 / 18 |
| l2_raw_gate | 16 / 32 | 16 / 32 | 16 / 32 |
| l2_ffn | 16 / 32 | 16 / 32 | 16 / 32 |

![介入比較](interventions.png)

[全350条件](metrics.csv)、[検証記録](summary.json)、[中間表現の変化](trace_comparisons.json)。
全条件のlogit/targetとJSONを保存。baselineの中間表現はNPZにも保存。
介入後の全中間表現はRAIDに保持し、[生データの場所とSHA256](raw_trace_manifest.json)を保存した。

## 確認後の解釈

bestのT49152の9桁目は、検証32例で16/32から、head8出力移植・RoPE位相のみ変更・第2層gate移植により32/32となった。
追加空白slot除外はlogitを変えず、改善しなかった。T49153の8桁目でも17/32→32/32を確認した。
逆方向にhead8の不良値を移植すると、良好T32768の9桁目は32/32→15/32へ悪化した。
finalの対象桁は元から全例正答。対象外の桁には誤りが残る。

[詳しい解釈と未解決事項](../../../logkv-readout-intervention.md)、[照合記録](review.json)。
