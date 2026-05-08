#!/usr/bin/env python3
"""实验7: 单核 L1 小表微基准 — 三阶段时间分解可视化"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os

font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if os.path.exists(font_path):
    font = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font.get_name()
    plt.rcParams['mathtext.fontset'] = 'dejavusans'

# 实验数据
table_labels = ['2×256\n(1KiB)', '4×256\n(2KiB)', '8×256\n(4KiB)', '16×256\n(8KiB)', '32×256\n(16KiB)']
n_lookups = [128, 64, 32, 16, 8]
total_us  = [0.1784, 0.1262, 0.1017, 0.0948, 0.0910]
scalar_us = [0.1151, 0.0680, 0.0464, 0.0407, 0.0393]
sve_acc_us = [0.0373, 0.0314, 0.0289, 0.0279, 0.0255]
sve_rdc_us = [0.0259, 0.0269, 0.0264, 0.0262, 0.0262]
l1_miss = [0.0, 0.0, 0.0, 0.0, 0.0]
per_lookup_ns = [1.39, 1.97, 3.18, 5.93, 11.38]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('实验 7：单核 L1 小表查表微基准 — 三阶段时间分解', fontsize=15, y=1.02)

# 颜色
colors = ['#3498DB', '#2ECC71', '#E74C3C']
hatches = ['', '///', '...']

# ── 图1: 总时间 vs 查表次数 ──
ax = axes[0, 0]
ax.bar(table_labels, total_us, color='#9B59B6', width=0.5, edgecolor='white')
for i, v in enumerate(total_us):
    ax.text(i, v + 0.003, f'{v:.4f}', ha='center', fontsize=9, fontweight='bold')
ax.set_ylabel('平均总耗时 (μs)', fontsize=11)
ax.set_title('总时间随表大小变化', fontsize=12)
ax.set_ylim(0, 0.22)

# ── 图2: 三阶段堆叠柱状图 ──
ax = axes[0, 1]
x = np.arange(len(table_labels))
width = 0.5
bottoms = np.zeros(len(table_labels))
for vals, label, color in [(scalar_us, '标量查表', colors[0]),
                            (sve_acc_us, 'SVE累加', colors[1]),
                            (sve_rdc_us, 'SVE归约', colors[2])]:
    bars = ax.bar(x, vals, width, bottom=bottoms, label=label, color=color, edgecolor='white')
    bottoms += vals
# 标注阶段占比
for i in range(len(table_labels)):
    h_scalar = scalar_us[i] / total_us[i] * 100
    h_sve = sve_acc_us[i] / total_us[i] * 100
    h_rdc = sve_rdc_us[i] / total_us[i] * 100
    ax.text(i, scalar_us[i]/2, f'{h_scalar:.0f}%', ha='center', fontsize=8, color='white', fontweight='bold')
    ax.text(i, scalar_us[i] + sve_acc_us[i]/2, f'{h_sve:.0f}%', ha='center', fontsize=8, fontweight='bold')
    ax.text(i, total_us[i] - sve_rdc_us[i]/2, f'{h_rdc:.0f}%', ha='center', fontsize=8, color='white', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(table_labels, fontsize=8)
ax.set_ylabel('平均耗时 (μs)', fontsize=11)
ax.set_title('三阶段时间分解与占比', fontsize=12)
ax.legend(fontsize=9, loc='upper left')

# ── 图3: 每次查表延迟 vs 查表次数 ──
ax = axes[0, 2]
ax2 = ax.twinx()
line1 = ax.plot(n_lookups, per_lookup_ns, 'o-', color='#E74C3C', linewidth=2, markersize=10, label='每次查表(ns)')
line2 = ax2.plot(n_lookups, total_us, 's--', color='#9B59B6', linewidth=2, markersize=8, label='总耗时(μs)')
ax.set_xlabel('查表次数', fontsize=11)
ax.set_ylabel('每次查表 (ns)', fontsize=11, color='#E74C3C')
ax2.set_ylabel('总耗时 (μs)', fontsize=11, color='#9B59B6')
ax.set_title('"每次查表"指标的失真效应', fontsize=12)
ax.invert_xaxis()
ax.set_xticks(n_lookups)
ax.tick_params(axis='y', labelcolor='#E74C3C')
ax2.tick_params(axis='y', labelcolor='#9B59B6')
lines = line1 + line2
ax.legend(lines, [l.get_label() for l in lines], fontsize=9, loc='center right')

# ── 图4: 三阶段时间 vs 查表次数（趋势线）──
ax = axes[1, 0]
ax.plot(n_lookups, [s * 1000 for s in scalar_us], 'o-', color=colors[0], linewidth=2, markersize=8, label='标量查表')
ax.plot(n_lookups, [s * 1000 for s in sve_acc_us], 's-', color=colors[1], linewidth=2, markersize=8, label='SVE累加')
ax.plot(n_lookups, [s * 1000 for s in sve_rdc_us], 'D-', color=colors[2], linewidth=2, markersize=8, label='SVE归约(恒定)')
ax.set_xlabel('查表次数', fontsize=11)
ax.set_ylabel('时间 (ns)', fontsize=11)
ax.set_title('各阶段时间随查表次数的变化', fontsize=12)
ax.invert_xaxis()
ax.set_xticks(n_lookups)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
# 标注归约恒定值
ax.axhline(y=26, color=colors[2], linestyle=':', alpha=0.5)
ax.text(40, 27, f'~26 ns 恒定', fontsize=8, color=colors[2])

# ── 图5: 固定开销 vs 可变开销分解 ──
ax = axes[1, 1]
# 固定开销 = 归约 + (clock_gettime × 3) 的基准
# 可变开销 = 标量 + SVE累加（与查表次数相关）
fixed = sve_rdc_us[:]  # 归约作为固定开销的代理
variable = [total_us[i] - fixed[i] for i in range(len(total_us))]
ax.bar(x, fixed, width, label='固定开销(归约+clock)', color='#E74C3C', edgecolor='white')
ax.bar(x, variable, width, bottom=fixed, label='可变开销(查表+累加)', color='#3498DB', edgecolor='white')
# 标注固定开销占比
for i in range(len(table_labels)):
    pct = fixed[i] / total_us[i] * 100
    ax.text(i, fixed[i]/2, f'{pct:.0f}%', ha='center', fontsize=9, color='white', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(table_labels, fontsize=8)
ax.set_ylabel('平均耗时 (μs)', fontsize=11)
ax.set_title('固定开销 vs 可变开销分解', fontsize=12)
ax.legend(fontsize=9)
for i, v in enumerate(total_us):
    ax.text(i, v + 0.003, f'{v:.4f}', ha='center', fontsize=8, color='#333')

# ── 图6: 结论要点 ──
ax = axes[1, 2]
ax.axis('off')
text = """
主要结论

1. L1 查表延迟 ~3-4 周期
   每次 BF16 L1 命中加载 ≈ 1.39 ns

2. L1-miss% = 0.00%（所有表大小）
   1~16 KiB 子表完全 L1 驻留

3. 标量查表占主导（~65%）
   SVE 累加 ~21%，归约 ~14%

4. "每次查表"指标在查表次数
   ≤32 时失真——固定开销主导

5. 归约时间恒定 ~26 ns
   验证三阶段计时插桩正确性

6. 实验 6 瓶颈确认：
   atomic 竞争 / MESI 流量
   而非 L1 查表本身
"""
ax.text(0.05, 0.95, text, fontsize=9, color='#333', va='top',
        transform=ax.transAxes, fontproperties=font,
        bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.8))

plt.tight_layout()
out_path = '/home/cuihaitao/cache_test/exp7_visualization.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'可视化已保存: {out_path}')
