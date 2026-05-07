#!/usr/bin/env python3
"""实验1-5：L2 Cache 优化 — 结果可视化"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os

# 中文字体
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if os.path.exists(font_path):
    font = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font.get_name()
    plt.rcParams['mathtext.fontset'] = 'dejavusans'

# 颜色方案
c_scalar = '#A23B72'
c_sve = '#2E86AB'
c_no_tile = '#A23B72'
c_tiled = '#2E86AB'
c_speedup = '#2E86AB'
c_efficiency = '#F18F01'
c_baseline_bar = '#E71D36'

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# ========== 图1: SVE 加速比 (Exp1) ==========
ax = axes[0, 0]
methods = ['Scalar\n标量', 'SVE\ngather']
gops_1 = [1.42, 1.63]
bars = ax.bar(methods, gops_1, color=[c_scalar, c_sve], width=0.5, edgecolor='white')
ax.set_ylabel('GOP/s', fontsize=11)
ax.set_title('实验1: SVE 向量化加速比 (单核)', fontsize=12)
for bar, v in zip(bars, gops_1):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
            f'{v:.2f}', ha='center', fontsize=11, fontweight='bold')
ax.set_ylim(0, 2.0)
# 标注加速比
ax.annotate('1.15×', xy=(0.5, 1.5), fontsize=10, color='#666',
            ha='center', fontweight='bold')

# ========== 图2: Tile 扫描热力图 (Exp2) ==========
ax = axes[0, 1]
ni_vals = [1, 2, 4, 8, 16, 32, 64, 128]
sj_vals = [8, 16, 32, 64, 128, 256, 512]
# GOP/s data from the results — best tile sizes
# Ni rows, Sj columns
tile_gops = [
    [55.66, 55.41, 51.43, 43.55, 34.69, 24.68, 14.41],  # Ni=1
    [56.83, 55.75, 51.65, 43.18, 34.79, 24.60, 14.67],  # Ni=2
    [56.64, 56.22, 51.89, 43.59, 34.67, 25.05, 14.90],  # Ni=4
    [56.54, 55.91, 52.39, 43.73, 34.61, 24.75, 14.42],  # Ni=8
    [56.47, 55.19, 52.00, 43.66, 35.41, 27.89, 14.29],  # Ni=16
    [57.33, 56.32, 52.56, 43.66, 35.63, 27.38, 14.41],  # Ni=32
    [56.84, 55.66, 52.18, 42.39, 35.82, 24.29, 13.85],  # Ni=64
    [56.40, 55.42, 51.93, 42.60, 34.36, 22.81, 11.34],  # Ni=128
]
im = ax.imshow(tile_gops, cmap='YlGnBu', aspect='auto', vmin=10, vmax=60)
ax.set_xticks(range(len(sj_vals)))
ax.set_xticklabels([str(s) for s in sj_vals])
ax.set_yticks(range(len(ni_vals)))
ax.set_yticklabels([str(n) for n in ni_vals])
ax.set_xlabel('Sj (B_T tile 宽度)', fontsize=11)
ax.set_ylabel('Ni (A tile 高度)', fontsize=11)
ax.set_title('实验2: Tile 尺寸扫描 GOP/s (80核)', fontsize=12)
# 标注最优
max_idx = np.unravel_index(np.argmax(tile_gops), np.shape(tile_gops))
ax.plot(max_idx[1], max_idx[0], 'r*', markersize=15, markeredgecolor='white', markeredgewidth=1)
ax.text(max_idx[1], max_idx[0] + 0.3, f'最优\nNi=32 Sj=8\n{np.max(tile_gops):.2f} GOP/s',
        ha='center', fontsize=8, color='red', fontweight='bold')
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('GOP/s', fontsize=9)

# ========== 图3: 单 NUMA 扩展 (Exp3) ==========
ax = axes[0, 2]
threads_3 = [1, 2, 4, 8, 16, 32, 64, 80]
speedup_3 = [1.0, 1.91, 3.76, 7.71, 14.96, 26.28, 21.84, 27.07]
efficiency_3 = [100, 95, 94, 96, 93, 82, 34, 34]
gops_3 = [1.60, 3.04, 6.00, 12.30, 23.86, 41.93, 34.84, 43.19]

color_speedup = '#2E86AB'
color_eff = '#F18F01'
ax2 = ax.twinx()
line1 = ax.plot(threads_3, speedup_3, 'o-', color=color_speedup, linewidth=2, markersize=8, label='加速比')
line2 = ax2.plot(threads_3, efficiency_3, 's--', color=color_eff, linewidth=2, markersize=7, label='效率 (%)')
ax.set_xlabel('线程数', fontsize=11)
ax.set_ylabel('加速比', fontsize=11, color=color_speedup)
ax2.set_ylabel('效率 (%)', fontsize=11, color=color_eff)
ax.set_title('实验3: 单 NUMA Node 多核扩展', fontsize=12)
ax.set_xticks(threads_3)
ax.tick_params(axis='y', labelcolor=color_speedup)
ax2.tick_params(axis='y', labelcolor=color_eff)
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc='center right', fontsize=9)
# 标注效率下降点
ax.annotate('效率下降\n带宽饱和', xy=(32, 26.28), xytext=(24, 22),
            arrowprops=dict(arrowstyle='->', color='gray'), fontsize=8, color='gray')
# 标注GOP/s
for i, (t, g) in enumerate(zip(threads_3, gops_3)):
    if t in [1, 8, 32, 80]:
        ax.annotate(f'{g} GOP/s', xy=(t, speedup_3[i]), xytext=(t, speedup_3[i]+2),
                    ha='center', fontsize=7, color='#555')

# ========== 图4: 跨 NUMA 扩展 (Exp4) ==========
ax = axes[1, 0]
numa_configs = ['1', '80\n(1 node)', '160\n(2 nodes)', '240\n(3 nodes)', '320\n(4 nodes)']
numa_gops = [1.56, 97.88, 166.55, 154.72, 208.06]
numa_speedup = [1.0, 62.65, 106.60, 99.03, 133.17]
numa_efficiency = [100, 78, 67, 41, 42]

bars = ax.bar(numa_configs, numa_gops, color=[c_sve if i > 0 else c_scalar for i in range(len(numa_configs))],
              width=0.6, edgecolor='white')
ax.set_ylabel('GOP/s', fontsize=11)
ax.set_title('实验4: 跨 NUMA 扩展 (Tile 32×128)', fontsize=12)
for bar, v, sp, eff in zip(bars, numa_gops, numa_speedup, numa_efficiency):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f'{v:.1f}', ha='center', fontsize=9, fontweight='bold')
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.5,
            f'{sp:.0f}×\n{eff:.0f}% eff', ha='center', fontsize=7, color='white', fontweight='bold')

# ========== 图5: 矩阵规模扩展 (Exp5) ==========
ax = axes[1, 1]
scales = ['1×\n256×1024', '2×\n512×2048', '4×\n1024×4096', '8×\n2048×8192']
gops_1t = [1.52, 1.62, 1.62, 1.61]
gops_80_no_tile = [42.40, 49.19, 52.46, 54.44]
gops_80_tiled = [43.36, 45.92, 56.23, 57.09]

x = np.arange(len(scales))
width = 0.25
bars1 = ax.bar(x - width, gops_1t, width, label='1线程 无tile', color=c_scalar, alpha=0.7)
bars2 = ax.bar(x, gops_80_no_tile, width, label='80线程 无tile', color=c_no_tile, alpha=0.7)
bars3 = ax.bar(x + width, gops_80_tiled, width, label='80线程 Tile 32×128', color=c_tiled, alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(scales, fontsize=8)
ax.set_ylabel('GOP/s', fontsize=11)
ax.set_title('实验5: 矩阵规模扩展 × Tile 优化 (80核)', fontsize=12)
ax.legend(fontsize=8)

# 标注 tile 收益
for i in range(4):
    delta = (gops_80_tiled[i] - gops_80_no_tile[i]) / gops_80_no_tile[i] * 100
    color = 'green' if delta > 0 else 'red'
    ax.annotate(f'{delta:+.0f}%', xy=(i + width/2, gops_80_tiled[i]),
                xytext=(i + width/2, gops_80_tiled[i] + 2), ha='center', fontsize=8,
                color=color, fontweight='bold')

# ========== 图6: 性能汇总对比 ==========
ax = axes[1, 2]
experiments = ['Exp1\n标量', 'Exp1\nSVE', 'Exp2\nTile最优', 'Exp3\n80核\n行并行',
               'Exp4\n320核\n跨NUMA', 'Exp5\n8× 无tile', 'Exp5\n8× 有tile']
all_gops = [1.42, 1.63, 57.33, 43.19, 208.06, 54.44, 57.09]
colors = [c_scalar, c_sve, c_tiled, c_no_tile, '#F18F01', '#A23B72', '#2E86AB']

bars = ax.bar(experiments, all_gops, color=colors, width=0.6, edgecolor='white')
ax.set_ylabel('GOP/s', fontsize=11)
ax.set_title('L2 Cache 优化 — 性能汇总', fontsize=12)
ax.set_xticklabels(experiments, fontsize=7)
for bar, v in zip(bars, all_gops):
    y_pos = bar.get_height() + 1
    ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{v:.1f}',
            ha='center', fontsize=8, fontweight='bold')

plt.suptitle('L2 Cache 优化实验 — 鲲鹏 SVE 80核 性能分析', fontsize=16, y=1.01)
plt.tight_layout()
out_path = '/home/cuihaitao/cache_test/exp1_5_visualization.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'可视化已保存: {out_path}')
