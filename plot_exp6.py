#!/usr/bin/env python3
"""实验6：L1 分组建表查表 — 结果可视化"""

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
else:
    # fallback
    for f in fm.findSystemFonts():
        if 'NotoSansCJK' in f and 'Regular' in f:
            font = fm.FontProperties(fname=f)
            plt.rcParams['font.family'] = font.get_name()
            break

# 数据
G = [1, 2, 4, 8, 16, 32, 64]
threads = [1, 2, 4, 8, 16, 32, 64]
subtable_kib = [128, 64, 32, 16, 8, 4, 2]
levels = ['L2', 'L1', 'L1', 'L1', 'L1', 'L1', 'L1']
compute_min = [28025.0, 19492.7, 7778.2, 3400.3, 1939.4, 2167.8, 1208.2]
compute_max = [28025.0, 19501.4, 8978.6, 6175.0, 4475.6, 3318.5, 2460.4]
sync_time = [1.6, 14.4, 1073.1, 2803.6, 2573.6, 460.3, 602.6]
total_time = [28026.7, 19515.8, 10051.7, 8978.6, 7049.1, 3778.8, 3063.1]
gops = [0.77, 1.10, 2.14, 2.39, 3.05, 5.69, 7.02]
baseline_gops = 18.89
baseline_total = 1137.8

# 计算加速比、效率
speedup = [total_time[0] / t for t in total_time]
efficiency = [s / t * 100 for s, t in zip(speedup, threads)]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 颜色方案
c_l1 = '#2E86AB'
c_l2 = '#A23B72'
c_sync = '#F18F01'
c_compute = '#2E86AB'
c_baseline = '#E71D36'

# --- 图1: GOP/s vs G ---
ax = axes[0, 0]
ax.bar([str(g) for g in G], gops, color=c_l1, width=0.6, label='分组查表')
ax.axhline(y=baseline_gops, color=c_baseline, linestyle='--', linewidth=1.5, label=f'基线 SVE ({baseline_gops} GOP/s)')
ax.set_xlabel('G (子表数)', fontsize=11)
ax.set_ylabel('GOP/s', fontsize=11)
ax.set_title('吞吐率 (GOP/s)', fontsize=13)
ax.legend(fontsize=9)
for i, v in enumerate(gops):
    ax.text(i, v + 0.3, f'{v:.2f}', ha='center', fontsize=9)

# --- 图2: 计算时间 (min~max) ---
ax = axes[0, 1]
x = np.arange(len(G))
width = 0.35
ax.bar(x - width/2, compute_min, width, label='min', color=c_l1, alpha=0.7)
ax.bar(x + width/2, compute_max, width, label='max', color=c_sync, alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels([str(g) for g in G])
ax.set_xlabel('G (子表数)', fontsize=11)
ax.set_ylabel('时间 (μs)', fontsize=11)
ax.set_title('计算耗时 min~max (μs)', fontsize=13)
ax.legend(fontsize=9)

# 标注不均衡度
for i in range(len(G)):
    ratio = compute_max[i] / compute_min[i]
    ax.text(i, compute_max[i] + 200, f'{ratio:.2f}×', ha='center', fontsize=8, color='#666')

# --- 图3: 同步时间 ---
ax = axes[0, 2]
bars = ax.bar([str(g) for g in G], sync_time, color=c_sync, width=0.6)
ax.set_xlabel('G (子表数)', fontsize=11)
ax.set_ylabel('同步时间 (μs)', fontsize=11)
ax.set_title('Atomic 同步开销 (μs)', fontsize=13)
for i, v in enumerate(sync_time):
    pct = v / total_time[i] * 100
    ax.text(i, v + 30, f'{v:.0f} ({pct:.0f}%)', ha='center', fontsize=8)

# --- 图4: 总耗时 ---
ax = axes[1, 0]
ax.bar([str(g) for g in G], compute_min, width, label='计算(min)', color=c_compute, alpha=0.6)
ax.bar([str(g) for g in G], [compute_max[i] - compute_min[i] for i in range(len(G))],
       bottom=compute_min, width=width, label='负载不均开销', color='#F5B042', alpha=0.6)
ax.bar([str(g) for g in G], sync_time, width=width,
       bottom=[compute_max[i] for i in range(len(G))],
       label='同步(atomic)', color=c_sync, alpha=0.8)
ax.axhline(y=baseline_total, color=c_baseline, linestyle='--', linewidth=1.5,
           label=f'基线总耗时 ({baseline_total} μs)')
ax.set_xlabel('G (子表数)', fontsize=11)
ax.set_ylabel('时间 (μs)', fontsize=11)
ax.set_title('总耗时构成分析 (μs)', fontsize=13)
ax.legend(fontsize=8)

# --- 图5: 加速比 vs 线程数 ---
ax = axes[1, 1]
ax.plot(threads, speedup, 'o-', color=c_l1, linewidth=2, markersize=8, label='分组查表')
ax.plot(threads, threads, '--', color='#999', linewidth=1, label='线性加速')
ax.set_xlabel('线程数 (= G)', fontsize=11)
ax.set_ylabel('加速比 (相对 G=1)', fontsize=11)
ax.set_title('加速比 vs 线程数', fontsize=13)
ax.legend(fontsize=9)
ax.set_xscale('log', base=2)
ax.set_xticks(threads)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

# --- 图6: 子表大小 vs GOP/s ---
ax = axes[1, 2]
sc = ax.scatter(subtable_kib, gops, c=threads, s=[t*20 for t in threads],
                cmap='viridis', alpha=0.8, edgecolors='k')
ax.set_xlabel('子表大小 (KiB)', fontsize=11)
ax.set_ylabel('GOP/s', fontsize=11)
ax.set_title('子表大小 vs 吞吐率', fontsize=13)
ax.invert_xaxis()  # 大G = 小表在右边
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('线程数 (G)', fontsize=9)
# 标注L1/L2分界
ax.axvline(x=64, color='red', linestyle=':', linewidth=1, alpha=0.7)
ax.text(65, max(gops)-0.5, 'L1d=64K', fontsize=9, color='red', alpha=0.7)

plt.suptitle('实验6：L1 分组建表查表 — 结果分析', fontsize=16, y=1.01)
plt.tight_layout()
out_path = '/home/cuihaitao/cache_test/exp6_visualization.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'可视化已保存: {out_path}')
