"""Render the current standard LogKV architecture as a 16:9 SVG and PNG.

Run from the repository root:
    .venv/bin/python doc/figures/render_logkv_overview.py

Requires matplotlib and Noto Sans CJK JP (installed on this development host).
Sources: models/logkv/{attention,modeling}.py, training/train_logkv.py.
The illustrated slots are the state BEFORE inserting token 27 (C=4).
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent
REGULAR = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
BOLD = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
BG, INK, MUTED, LINE = '#F4F6F8', '#162C40', '#5C6F7F', '#DCE4EB'
TEAL, BLUE, PURPLE, ORANGE = '#087E8B', '#2E63C7', '#7655AF', '#B8651B'
TEAL_BG, BLUE_BG, PURPLE_BG, ORANGE_BG = '#E8F5F5', '#EBF1FC', '#F0EBF8', '#FFF2E3'


def main():
    plt.rcParams['svg.fonttype'] = 'path'  # Portable Japanese text, no font dependency in viewers.
    plt.rcParams['svg.hashsalt'] = 'logkv-overview'
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1920), ylim=(1080, 0))
    ax.set_axis_off()

    def box(x, y, w, h, fill='white', edge=LINE, radius=18, lw=1):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle=f'round,pad=0,rounding_size={radius}',
                     facecolor=fill, edgecolor=edge, linewidth=lw, zorder=2))

    def text(x, y, value, size=24, color=INK, bold=False, align='left', **kwargs):
        ax.text(x, y, value, fontsize=size * 72 / 120,
                fontproperties=BOLD if bold else REGULAR, color=color,
                ha=align, va='center', zorder=5, **kwargs)

    def arrow(x1, y1, x2, y2, color=MUTED, width=1.5, style='-|>'):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                     mutation_scale=12, linewidth=width, color=color,
                     shrinkA=0, shrinkB=0, zorder=3))

    def line(points, color=LINE, width=1.2, **kwargs):
        ax.plot(*zip(*points), color=color, linewidth=width, zorder=3, **kwargs)

    def panel(x, w, number, title):
        box(x, 224, w, 656)
        box(x + 24, 248, 38, 34, INK, INK, radius=9)
        text(x + 43, 265, number, 19, 'white', True, 'center')
        text(x + 76, 265, title, 26, bold=True)

    # Header.
    box(64, 48, 7, 112, TEAL, TEAL, radius=3)
    text(92, 79, 'LogKV', 59, bold=True)
    text(303, 85, '長い履歴を、小さな階層メモリへ', 43, bold=True)
    text(92, 143, '直近は細かく、古い履歴はまとめて参照。系列が伸びても、キャッシュは対数サイズ。', 25, MUTED)
    text(1856, 187, '標準学習構成  /  C = 4  /  位置埋め込みなし', 19, MUTED, align='right')

    # 01 — end-to-end model, with the internal order of one block.
    panel(64, 340, '01', 'トークンを処理する')
    for y, label, fill, color in [(312, '入力トークン', BG, INK),
                                  (373, 'Embedding', BG, INK)]:
        box(94, y, 280, 42, fill, LINE, radius=10)
        text(234, y + 21, label, 23, color, True, 'center')
    arrow(234, 356, 234, 369)
    arrow(234, 417, 234, 442)
    box(84, 447, 300, 275, '#F7F9FC', LINE, radius=14)
    text(102, 468, 'LogKV Block', 19, bold=True)
    text(365, 468, '× N', 21, BLUE, True, 'right')
    for y, title, subtitle, fill, color in [
        (491, 'CausalConv + 残差', '幅4で近傍の並びを捉える', TEAL_BG, TEAL),
        (563, 'LogKV Attention + 残差', '階層を参照 → gate → 出力射影', BLUE_BG, BLUE),
        (635, 'SwiGLU FFN + 残差', '特徴を変換する', BG, INK),
    ]:
        box(98, y, 272, 58, fill, fill, radius=10)
        text(234, y + 18, title, 19, color, True, 'center')
        text(234, y + 42, subtitle, 16, MUTED, align='center')
    arrow(234, 550, 234, 560)
    arrow(234, 622, 234, 632)
    text(234, 707, '各枝の前にRMSNorm', 16, MUTED, align='center')
    arrow(234, 724, 234, 740)
    box(94, 745, 280, 44, BG, LINE, radius=10)
    text(234, 767, 'RMSNorm → Linear', 22, bold=True, align='center')
    arrow(234, 791, 234, 807)
    box(94, 812, 280, 42, INK, INK, radius=10)
    text(234, 833, '次トークンのlogits', 21, 'white', True, 'center')

    # 02 — chronological coverage. Width encodes the number of source tokens,
    # NOT physical cache storage: every colored rectangle is one KV slot.
    panel(428, 888, '02', '必要な粒度で、過去全体を読む')
    text(454, 311, '例：位置27を読む時（挿入前）', 23, bold=True)
    text(454, 343, '色付きの箱1個 = 1 KVスロット。横幅は担当するトークン区間。', 19, MUTED)
    start, pitch = 550, 26
    arrow(start, 382, start + 28*pitch, 382, LINE)
    text(start, 371, '古い', 16, MUTED)
    text(start + 28*pitch, 371, '現在', 16, MUTED, align='right')
    for i in [0, 16, 20, 24, 27, 28]:
        line([(start + pitch*i, 398), (start + pitch*i, 665)], '#E8EDF2', .8, linestyle=(0, (3, 4)))
    rows = [
        (422, 'L2', '16個分', PURPLE, PURPLE_BG, [(0, 16, '0–15')]),
        (496, 'L1', '4個分', BLUE, BLUE_BG, [(16, 4, '16–19'), (20, 4, '20–23')]),
        (570, 'L0', '1個分', TEAL, TEAL_BG, [(24, 1, '24'), (25, 1, '25'), (26, 1, '26')]),
        (644, 'Self', '現在', ORANGE, ORANGE_BG, [(27, 1, '27')]),
    ]
    for y, level, caption, color, fill, slots in rows:
        text(454, y - 7, level, 24, color, True)
        text(454, y + 17, caption, 16, MUTED)
        for i, width, label in slots:
            x = start + pitch*i
            box(x + 1, y - 23, pitch*width - 3, 46, fill, color, radius=6, lw=1.2)
            label_x = x + pitch*width - 43 if width == 16 else x + pitch*width/2 - .5
            text(label_x, y, label, 21 if width > 1 else 16,
                 color, True, 'center')
    text(565, 425, '過去16トークンを1個に要約', 18, PURPLE)
    text(550, 689, '担当区間は重複なし：過去27トークン → 6スロット ＋ Self 1個', 20, bold=True)
    arrow(895, 706, 895, 724, BLUE)
    box(454, 732, 836, 111, BLUE_BG, BLUE_BG, radius=14)
    text(478, 757, '現在のQ × 全階層のK → 単一softmax → Vの加重和', 23, BLUE, True)
    text(478, 798, r'$\mathrm{logit}_i = q \cdot k_i / \sqrt{d_h} - i\,\log C$', 27, BLUE)
    text(1266, 819, 'i：圧縮レベル  /  Selfの減衰は0', 17, MUTED, align='right')

    # 03 — recurrence / compression, using the same concrete example.
    panel(1340, 516, '03', '4個そろったら、1個へ')
    text(1366, 311, '読み出した後、現在トークンを追加。', 22, bold=True)
    text(1366, 343, '同じ階層で完成したチャンクを上位へ送る。', 19, MUTED)
    for j, label in enumerate(['24', '25', '26', '27']):
        x = 1368 + j*119
        fill, color = (ORANGE_BG, ORANGE) if j == 3 else (TEAL_BG, TEAL)
        box(x, 391, 104, 50, fill, color, radius=10)
        text(x + 52, 416, label, 25, color, True, 'center')
        line([(x + 52, 444), (x + 52, 464), (1598, 464)], MUTED)
    arrow(1598, 464, 1598, 487, BLUE)
    box(1438, 494, 320, 55, BLUE_BG, BLUE, radius=12)
    text(1598, 521, 'L1の要約 24–27', 25, BLUE, True, 'center')
    text(1366, 583, 'Compressor：内容に応じた重み付き平均', 21, bold=True)
    text(1366, 619, 'チャンク末尾のQで、4個のKを見る。', 21, MUTED)
    text(1366, 657, r'$a = \mathrm{softmax}(q_{\mathrm{last}} K^\top / \sqrt{d_h})$', 27, BLUE)
    text(1366, 699, r'$\bar{K} = aK,\quad \bar{V} = aV,\quad \bar{q} = q_{\mathrm{last}}$', 28, BLUE)
    box(1366, 738, 464, 108, BG, BG, radius=12)
    text(1384, 762, '元の4個は下位から除き、要約だけを保持。', 20, bold=True)
    text(1384, 797, '上位も4個そろえば、さらに圧縮。', 20, MUTED)
    text(1384, 827, '各レベルに残すのは最大3個（C − 1）。', 20, MUTED)

    # Takeaways and precise storage scope.
    box(64, 905, 1792, 110, INK, INK, radius=18)
    for x, kicker, value, detail in [
        (90, '階層メモリ', '系列長に対して O(log L)', '全履歴を区間でカバーし、要約として保持'),
        (708, '各層の逐次推論状態量', r'$O(C\,\log_C L\,d)$', '未完成Q/K/V ＋ 位置 ＋ Convの短い履歴'),
        (1306, '連続して読み書き', '任意長の分割入力に対応', '状態を引き継ぐstep ／ 1トークン専用predict'),
    ]:
        text(x, 926, kicker, 16, '#9FBECC')
        text(x, 958, value, 29, 'white', True)
        text(x, 990, detail, 18, '#C8D8E2')
    text(64, 1045, '圧縮は情報の要約。L：系列長、d：隠れ次元、dh：ヘッド次元。状態量は逐次推論時の値。', 17, MUTED)
    text(1856, 1045, '標準：Conv4・gate・Selfあり ／ 固定レベル減衰', 17, MUTED, align='right')

    metadata = {'Title': 'LogKV architecture overview — current standard configuration', 'Date': None,
                'Description': '16:9 Japanese overview: model blocks, non-overlapping interval slots at C=4 and t=27, and recursive attention pooling.'}
    fig.savefig(OUT / 'logkv-overview.svg', metadata=metadata)
    fig.savefig(OUT / 'logkv-overview.png', dpi=240, metadata={'Title': metadata['Title']})
    plt.close(fig)
    print('Created doc/figures/logkv-overview.svg and logkv-overview.png (3840 × 2160).')


if __name__ == '__main__':
    main()
