#!/usr/bin/env python3
"""实验8：两级查表优化原理可视化"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams['font.family'] = 'WenQuanYi Micro Hei'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['hatch.linewidth'] = 0.5

FIG_W, FIG_H = 18, 10


def draw_panel_a(ax):
    """Panel (a): 预处理 — 全表切分为 G 个 float 子表 + A 矩阵分组"""
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 8)
    ax.set_title("(a) 预处理：全 LUT → G 个 float 细表 + A 矩阵按 idxA 分组", fontsize=13, fontweight='bold', loc='left')

    # ── 全 LUT ──
    ax.text(1, 6.5, "完整 LUT\n256×256 float\n= 256 KiB (L2)", fontsize=9, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#2b83ba', alpha=0.7, edgecolor='none'))

    # 箭头：LUT 切分
    for gi in range(4):
        x = 4.5 + gi * 2.2
        y = 4
        w, h = 1.8, 2.2
        color = plt.cm.Set2(gi / 4)
        rect = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                              boxstyle="round,pad=0.1", facecolor=color, edgecolor='gray', linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x, y + 0.3, f"细表 {gi}", fontsize=8, ha='center', va='center', fontweight='bold')
        ax.text(x, y - 0.2, f"step×256 float", fontsize=7, ha='center', va='center', color='gray')
        ax.text(x, y - 0.7, f"G≥8: ≤32 KiB", fontsize=7, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='#ffffbf', alpha=0.6, edgecolor='none'))

    # 切分箭头
    ax.annotate('', xy=(3.2, 5.5), xytext=(4.2, 5.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.text(3.7, 5.8, "按 idxA\n值域切分", fontsize=7, ha='center', va='bottom', color='gray')

    # ── A 矩阵分组（左下） ──
    ax.text(0.3, 1.5, "A 矩阵\n(N×L)", fontsize=9, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fdae61', alpha=0.6, edgecolor='none'))

    # 分组后示意
    for gi in range(4):
        x = 4.5 + gi * 2.2
        y = 1.0
        w2, h2 = 1.8, 1.0
        color = plt.cm.Set2(gi / 4)
        rect = FancyBboxPatch((x - w2 / 2, y - h2 / 2), w2, h2,
                              boxstyle="round,pad=0.05", facecolor=color, edgecolor='gray', linewidth=0.8, alpha=0.5)
        ax.add_patch(rect)
        ax.text(x, y, f"组{gi} idxA", fontsize=7, ha='center', va='center')
        if gi == 3:
            ax.text(x + w2 / 2 + 0.3, y, "row_start\nrow_count", fontsize=7, ha='left', va='center', color='gray')

    ax.annotate('', xy=(2.1, 2.0), xytext=(3.8, 1.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2, linestyle='dashed'))

    # ── 粗表（右上角） ──
    coarse_x = 14
    ax.text(coarse_x, 6.5, "粗表\n(G 个指针)\n< 1 KiB", fontsize=9, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#abdda4', alpha=0.8, edgecolor='none'))
    for gi in range(4):
        x = coarse_x + 2.2
        y = 5.5 - gi * 0.8
        ax.plot([coarse_x + 0.6, x - 0.3], [6.3 - gi * 0.3, y], color='gray', linewidth=0.6, linestyle=':')
        ax.text(x, y, f"→ 细表{gi}", fontsize=7, ha='left', va='center', color='gray')

    # 图例
    ax.text(0.3, 7.3, "细表 = float (4B/entry), 直接 LUT 复制, 无 BF16 转换", fontsize=8, color='gray', ha='left')
    ax.text(0.3, 6.9, "粗表 = 细表指针数组, G 个条目, 常驻 L1", fontsize=8, color='gray', ha='left')
    ax.axis('off')


def draw_panel_b(ax):
    """Panel (b): 运行时 — 80 核行并行 + 两级查表"""
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.set_title("(b) 运行时：80 核行并行 + 两级查表（无 atomic）", fontsize=13, fontweight='bold', loc='left')

    # C 矩阵输出
    for ri in range(10):
        for ci in range(4):
            x = 0.5 + ci * 0.55
            y = 9.0 - ri * 0.6
            color = plt.cm.tab10(ri % 10) if ri < 10 else 'gray'
            rect = plt.Rectangle((x, y), 0.45, 0.45, facecolor=color, alpha=0.4 + 0.3 * (ri % 2),
                                 edgecolor='gray', linewidth=0.3)
            ax.add_patch(rect)

    # 标签
    ax.text(0.5 + 4 * 0.55 + 0.3, 8.5, "C[i][j]", fontsize=8, color='gray', ha='left', va='center')
    ax.annotate('', xy=(0.5 + 4 * 0.55 + 0.1, 9.0 - 0 * 0.6 + 0.2),
                xytext=(0.5 + 4 * 0.55 + 0.1, 9.0 - 9 * 0.6 + 0.2),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=0.8))
    ax.text(0.5 + 4 * 0.55 + 0.5, 5.5, "N=128\n行", fontsize=7, color='gray', ha='left', va='center')

    ax.text(0.5, 9.6, "输出\nC 矩阵", fontsize=8, ha='center', va='bottom', color='gray')

    # C 矩阵标注：每行由不同核处理
    for ri in range(0, 10, 2):
        core_id = (ri * 8) % 80
        x = 0.5 + 4 * 0.55 + 0.6
        y = 9.0 - ri * 0.6 + 0.2
        ax.text(x, y, f"核{core_id}", fontsize=5, ha='left', va='center', color='gray', family='WenQuanYi Micro Hei')

    # ── 核的放大视图（右侧） ──
    # 一个核心的内部处理流程
    core_box_x = 6.5
    core_box_y = 2.0
    core_w = 11
    core_h = 6.5

    rect = FancyBboxPatch((core_box_x, core_box_y), core_w, core_h,
                          boxstyle="round,pad=0.2", facecolor='#f7f7f7', edgecolor='#333', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(core_box_x + core_w / 2, core_box_y + core_h - 0.2, "单个核心内部处理流",
            fontsize=10, ha='center', va='top', fontweight='bold')

    # 3 个子区域：粗表→细表、标量阶段、SVE gather 阶段
    # ① 粗表 → 细表选择
    ax.text(core_box_x + 1.0, core_box_y + 4.8, "① 粗表查索引", fontsize=8, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#abdda4', alpha=0.6, edgecolor='none'))
    ax.text(core_box_x + 1.0, core_box_y + 4.3, "g = idxA/(256/G)\nfine = fine_tables[g]",
            fontsize=7, ha='center', color='#333')
    ax.text(core_box_x + 1.0, core_box_y + 3.9, "→ 细表常驻 L1 ✅", fontsize=7, ha='center',
            color='green', fontweight='bold')

    # 箭头
    ax.annotate('', xy=(core_box_x + 2.3, core_box_y + 4.0),
                xytext=(core_box_x + 3.5, core_box_y + 4.0),
                arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))

    # ② 标量阶段：B_T 收集 + 索引计算
    ax.text(core_box_x + 4.5, core_box_y + 4.8, "② 标量阶段", fontsize=8, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#fdae61', alpha=0.5, edgecolor='none'))
    ax.text(core_box_x + 4.5, core_box_y + 4.3, "b_val = B_T[j][k_ptr[t]]\nidx = (a - base)×256 + b",
            fontsize=7, ha='center', color='#333')
    ax.text(core_box_x + 4.5, core_box_y + 3.9, "收集 B_T + 计算子表索引", fontsize=7, ha='center', color='gray')

    ax.annotate('', xy=(core_box_x + 6.0, core_box_y + 4.0),
                xytext=(core_box_x + 7.0, core_box_y + 4.0),
                arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))

    # ③ SVE gather + 累加
    ax.text(core_box_x + 8.5, core_box_y + 4.8, "③ SVE gather", fontsize=8, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#2b83ba', alpha=0.3, edgecolor='none'))
    ax.text(core_box_x + 8.5, core_box_y + 4.3,
            "svld1_gather(fine, idx)\nsvadd_f32_m → svaddv_f32",
            fontsize=7, ha='center', color='#333')
    ax.text(core_box_x + 8.5, core_box_y + 3.9, "→ C[i][j] += sum (无 atomic!)", fontsize=7,
            ha='center', color='green', fontweight='bold')

    # ── 不同 G 值细表 L1 驻留示意 ──
    ax.text(core_box_x + 0.5, core_box_y + 2.8, "细表大小 & L1 驻留:", fontsize=8, fontweight='bold')
    for gi, (g, size, lvl) in enumerate([("G=1", "256 KiB", "L2"), ("G=2", "128 KiB", "L2"),
                                          ("G=4", "64 KiB", "边界"), ("G=8", "32 KiB", "✅L1"),
                                          ("G=16", "16 KiB", "✅L1"), ("G=32", "8 KiB", "✅L1"),
                                          ("G=64", "4 KiB", "✅L1")]):
        x = core_box_x + 0.5 + gi * 1.5
        y = core_box_y + 2.1
        color = '#ffffbf' if '✅' in lvl else '#fee090'
        rect = FancyBboxPatch((x - 0.65, y - 0.25), 1.35, 0.5,
                              boxstyle="round,pad=0.05", facecolor=color, edgecolor='gray', linewidth=0.5)
        ax.add_patch(rect)
        ax.text(x, y + 0.05, f"{g} {size}", fontsize=6, ha='center', va='center')

    # ── 与实验 6 对比 ──
    ax.text(core_box_x + 0.5, core_box_y + 1.3, "vs 实验 6:", fontsize=8, fontweight='bold')
    ax.text(core_box_x + 0.5, core_box_y + 0.8, "  实验6: atomic合并 → 缓存行竞争瓶颈", fontsize=7, color='#d7191c')
    ax.text(core_box_x + 0.5, core_box_y + 0.45, "  实验8: 行并行 → 无 atomic, 满80核 ✅", fontsize=7, color='#1a9641')
    ax.text(core_box_x + 0.5, core_box_y + 0.1, "  实验8: float 子表 → 无 BF16→float 转换 ✅", fontsize=7, color='#1a9641')

    ax.axis('off')


def draw_panel_c(ax):
    """Panel (c): 伪代码对照"""
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 5.5)
    ax.set_title("(c) 核函数伪代码", fontsize=13, fontweight='bold', loc='left')

    code = (
        "#pragma omp parallel for schedule(static)        ← 80 核行并行\n"
        "for i in 0..N-1:                                 ← 每核分不同行\n"
        "    for j in 0..S-1:\n"
        "        sum = 0\n"
        "        for g in 0..G-1:                         ← 两级查表\n"
        "            cnt = row_count[i*G+g]\n"
        "            if cnt == 0: continue\n"
        "            fine = fine_tables[g]                  ← 粗表→细表 (L1 ✅)\n"
        "\n"
        "            # 阶段1: 标量  ←  B_T收集 + 子表索引计算\n"
        "            for t in 0..cnt:\n"
        "                idx[t] = (a - base)*256 + B_T[j][k[t]]\n"
        "\n"
        "            # 阶段2: SVE  ←  gather + 累加\n"
        "            acc = svadd_f32_m(acc,\n"
        "                  svld1_gather_u32index_f32(fine, idx))\n"
        "            sum += svaddv_f32(acc)\n"
        "\n"
        "        C[i][j] = sum         ← 无 atomic! 直接写\n"
    )

    ax.text(0.5, 3.0, code, fontsize=7.5, fontfamily='WenQuanYi Micro Hei', ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#fafafa', edgecolor='#ccc', linewidth=1))

    # 标注
    ax.annotate('行并行\n无竞争', xy=(3.2, 4.8), xytext=(4.5, 5.0),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.2), fontsize=7, color='green', fontweight='bold')
    ax.annotate('细表 L1\n常驻命中', xy=(9.5, 3.7), xytext=(11, 4.2),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.2), fontsize=7, color='blue', fontweight='bold')
    ax.annotate('SVE gather\n向量化查表', xy=(12, 2.0), xytext=(13.5, 2.5),
                arrowprops=dict(arrowstyle='->', color='#d7191c', lw=1.2), fontsize=7, color='#d7191c', fontweight='bold')
    ax.annotate('无 atomic\n直写 C', xy=(14, 1.0), xytext=(15.5, 1.4),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.2), fontsize=7, color='green', fontweight='bold')

    ax.axis('off')


# ── 主图 ──
fig = plt.figure(figsize=(FIG_W, FIG_H))

ax1 = fig.add_axes([0.04, 0.56, 0.92, 0.40])
ax2 = fig.add_axes([0.04, 0.06, 0.92, 0.48])
ax3 = fig.add_axes([0.04, 0.01, 0.92, 0.04])
ax3.axis('off')
ax3.text(0, 0, "平台: HiSilicon Kunpeng 256-bit SVE | L1d=64KiB | 80 cores | float 子表 (4B/entry) | 无 BF16 转换 | 无 atomic",
         fontsize=8, color='gray', ha='left', va='bottom')

draw_panel_a(ax1)
draw_panel_b(ax2)
draw_panel_c(ax2)

fig.suptitle("实验 8：两级查表优化原理 (float 子表 + 行并行 + SVE gather)",
             fontsize=15, fontweight='bold', y=0.97)

plt.savefig('exp8_visualization.png', dpi=200, bbox_inches='tight', facecolor='white')
print("已生成 exp8_visualization.png")
