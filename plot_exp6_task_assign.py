#!/usr/bin/env python3
"""实验6 分组建表查表 — 核任务分配可视化"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import numpy as np
import os

font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if os.path.exists(font_path):
    font = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font.get_name()
    plt.rcParams['mathtext.fontset'] = 'dejavusans'

fig = plt.figure(figsize=(15, 11))
fig.suptitle('实验6：L1 分组建表查表 — 核任务分配原理 (G=4 示例)', fontsize=16, y=0.98)

# ===== 颜色方案 =====
c_g0 = '#E74C3C'
c_g1 = '#3498DB'
c_g2 = '#2ECC71'
c_g3 = '#F39C12'
c_gray = '#E8E8E8'
c_atomic = '#8E44AD'
gc = [c_g0, c_g1, c_g2, c_g3]

# ====================================================================
# 图1: idxA 值域分组 → G 个子表 (左上)
# ====================================================================
ax1 = fig.add_axes([0.06, 0.55, 0.42, 0.35])
ax1.set_title('(a) idxA 值域分组 -> G 个子表', fontsize=12, fontweight='bold', pad=8)
ax1.axis('off')

# 完整 LUT
ax1.annotate('', xy=(3.5, 2.5), xytext=(3.5, 1.8),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='#666'))
rect = mpatches.FancyBboxPatch((0.5, 2.5), 6, 1.0, boxstyle="round,pad=0.1",
                                facecolor=c_gray, edgecolor='#999', lw=1.5)
ax1.add_patch(rect)
ax1.text(3.5, 3.0, '完整 LUT: 256x256 float = 256 KiB (L2)', ha='center', fontsize=9)

# 切分说明
ax1.text(3.5, 1.6, '按 idxA 范围均分 G 段, 每段转 BF16 (2B/entry)', ha='center', fontsize=8, color='#666')

# 4 个子表
step = 64
G = 4
for g in range(G):
    x0 = 0.5 + g * 1.5
    rect = mpatches.FancyBboxPatch((x0, 0.5), 1.4, 0.8,
                                    boxstyle="round,pad=0.08",
                                    facecolor=gc[g], alpha=0.75, ec='white', lw=1)
    ax1.add_patch(rect)
    lo = g * step
    hi = (g+1) * step - 1
    ax1.text(x0+0.7, 0.9, f'子表 {g}', ha='center', fontsize=9, color='white', fontweight='bold')
    ax1.text(x0+0.7, 0.65, f'idxA in [{lo},{hi}]', ha='center', fontsize=7, color='white')
    ax1.text(x0+0.7, 0.3, f'{step}x256 BF16\n{step*256*2//1024} KiB', ha='center', fontsize=7, color=gc[g])

# L1 标注
ax1.text(0.5+1.5*2-0.75, 0.1, 'L1 边界' if 1 else '', ha='center', fontsize=8, color='#666')
for g in range(G):
    x0 = 0.5 + g * 1.5
    label = 'L1 OK' if g >= 1 else 'L1 边界(64K)'
    color = 'green' if g >= 1 else '#CC8800'
    ax1.text(x0+0.7, -0.1, label, ha='center', fontsize=7, color=color, fontweight='bold')

ax1.set_xlim(0, 7.5)
ax1.set_ylim(-0.5, 4)

# ====================================================================
# 图2: 预处理 — A 每行按组分排 (右上)
# ====================================================================
ax2 = fig.add_axes([0.55, 0.55, 0.42, 0.35])
ax2.set_title('(b) 预处理: A 每行按 idxA 分组重排连续存储', fontsize=12, fontweight='bold', pad=8)
ax2.axis('off')

row_len = 16
# 生成伪随机数据
rng = np.random.RandomState(42)
a_vals = [int(rng.randint(0, 256)) for _ in range(row_len)]
groups = [v // 64 for v in a_vals]

# 原始 A[i] 行
y_a = 3.5
for k in range(row_len):
    rect = mpatches.FancyBboxPatch((k*0.4, y_a), 0.35, 0.45,
                                    boxstyle="round,pad=0.02",
                                    facecolor=gc[groups[k]], alpha=0.65, ec='white')
    ax2.add_patch(rect)
    ax2.text(k*0.4+0.175, y_a+0.225, str(a_vals[k]), ha='center', va='center', fontsize=6, color='white')
ax2.text(-1.0, y_a+0.225, 'A[i] 行:', fontsize=9, va='center', fontweight='bold')
# 标注 k 索引
for k in range(row_len):
    ax2.text(k*0.4+0.175, y_a-0.15, f'k{k}', ha='center', fontsize=4.5, color='#888')

# 箭头
ax2.annotate('', xy=(3.5, 2.7), xytext=(3.5, 3.2),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='#666'))
ax2.text(4.5, 2.95, '按 g = idxA/64 分组\n重排 + 记录 start/count', fontsize=7, color='#666', va='center')

# 重排后
y_g = 1.6
# 对 a_vals 重排
grouped = []
for g in range(4):
    for k in range(row_len):
        if groups[k] == g:
            grouped.append((k, a_vals[k], g))
# 画组分隔线
for g in range(1, 4):
    x_sep = g * 4 * 0.4
    ax2.axvline(x_sep - 0.025, ymin=0.08, ymax=0.32, color='#999', lw=0.8, ls='--')

for pos, (k, v, g) in enumerate(grouped):
    rect = mpatches.FancyBboxPatch((pos*0.4, y_g), 0.35, 0.45,
                                    boxstyle="round,pad=0.02",
                                    facecolor=gc[g], alpha=0.65, ec='white')
    ax2.add_patch(rect)
    ax2.text(pos*0.4+0.175, y_g+0.225, str(v), ha='center', va='center', fontsize=6, color='white')

ax2.text(-1.0, y_g+0.225, 'A_grouped:', fontsize=9, va='center', fontweight='bold')

# row_start / row_count
for g in range(4):
    xs = g * 4 * 0.4 + 0.6
    ax2.text(xs, y_g-0.25, f'组{g}: start={g*4}, count=4', ha='center', fontsize=6, color=gc[g])

# 说明框
ax2.text(5.5, 0.6, '数据结构:', fontsize=8, fontweight='bold')
ax2.text(5.5, 0.3, 'A_grouped[k]:  重排后的 idxA', fontsize=7)
ax2.text(5.5, 0.05, 'k_indices[k]:  原始 k 索引', fontsize=7)
ax2.text(5.5, -0.2, 'row_start[i*G+g]:  组起始位置', fontsize=7)
ax2.text(5.5, -0.45, 'row_count[i*G+g]:  组元素数', fontsize=7)

ax2.set_xlim(-1.5, 8)
ax2.set_ylim(-1.0, 4.5)

# ====================================================================
# 图3: G 核协作计算同一 C[i][j] (左下)
# ====================================================================
ax3 = fig.add_axes([0.06, 0.05, 0.55, 0.42])
ax3.set_title('(c) G=4 核协作计算同一个 C[i][j]', fontsize=12, fontweight='bold', pad=8)
ax3.axis('off')

# 左边：4 个核并行
y0 = 4.2
col_left = 0.5
core_w = 2.8
core_h = 0.85
step = 64

for g in range(4):
    y = y0 - g * 0.95
    # 核框
    rect = mpatches.FancyBboxPatch((col_left, y), core_w, core_h,
                                    boxstyle="round,pad=0.08",
                                    facecolor=gc[g], alpha=0.1, ec=gc[g], lw=2)
    ax3.add_patch(rect)
    ax3.text(col_left+core_h/2, y+core_h/2, str(g), ha='center', va='center',
             fontsize=14, color=gc[g], fontweight='bold',
             bbox=dict(boxstyle='circle', facecolor='white', ec=gc[g], lw=1.5))
    lo = g * step
    hi = (g+1)*step - 1
    ax3.text(col_left+0.5, y+core_h-0.15, f'子表{g} (idxA=[{lo},{hi}])', fontsize=7, color=gc[g], fontweight='bold')

    # 循环伪代码
    code_y = y + 0.15
    ax3.text(col_left+0.5, code_y+0.25,
             f'for i in 0..N-1:  for j in 0..S-1:', fontsize=6, color='#666')
    ax3.text(col_left+0.7, code_y,
             f'cnt = row_count[i*G+{g}]', fontsize=6, color=gc[g], fontweight='bold')
    ax3.text(col_left+0.7, code_y-0.25,
             f'for t: 查 sub{g} + SVE acc -> part_sum_{g}', fontsize=6, color='#666')

    # atomic 箭头到右边
    ax3.annotate('', xy=(6.5, y+core_h/2), xytext=(col_left+core_w+0.1, y+core_h/2),
                arrowprops=dict(arrowstyle='->', lw=1.5, color=c_atomic))

# 右边：C[i][j]
rect_c = mpatches.FancyBboxPatch((6.5, 1.2), 2.2, 2.5,
                                  boxstyle="round,pad=0.15",
                                  facecolor=c_atomic, alpha=0.1, ec=c_atomic, lw=2.5)
ax3.add_patch(rect_c)
ax3.text(7.6, 2.8, 'C[i][j]', ha='center', fontsize=13, fontweight='bold', color=c_atomic)
ax3.text(7.6, 2.3, 'part_sum_0', ha='center', fontsize=7, color=c_g0)
ax3.text(7.6, 2.0, '+ part_sum_1', ha='center', fontsize=7, color=c_g1)
ax3.text(7.6, 1.7, '+ part_sum_2', ha='center', fontsize=7, color=c_g2)
ax3.text(7.6, 1.4, '+ part_sum_3', ha='center', fontsize=7, color=c_g3)
ax3.text(7.6, 1.05, '#pragma omp atomic', ha='center', fontsize=7, color=c_atomic, fontweight='bold')

# B_T 行标注
ax3.text(col_left+0.2, 4.7, 'B_T[j] 行: 所有核共享读取', fontsize=8, color='#666', fontweight='bold')

# A_grouped 标注
ax3.text(col_left+0.2, -0.2, '核 g 从 A_grouped 读自己的部分 + 从 k_indices 获取原始 k',
         fontsize=7, color='#666')

# atomic 竞争说明
ax3.text(5.5, -0.6, 'Atomic 竞争: G 核同时 atomic 写同一 C[i][j], 触发 MESI 缓存行传递',
         fontsize=8, ha='center', color='#E74C3C', style='italic',
         bbox=dict(boxstyle='round', facecolor='#FFF3CD', ec='#E74C3C'))

ax3.set_xlim(0, 9.5)
ax3.set_ylim(-1.0, 5.5)

# ====================================================================
# 图4: 时序图 — 负载不均与同步开销 (右下)
# ====================================================================
ax4 = fig.add_axes([0.67, 0.05, 0.3, 0.42])
ax4.set_title('(d) 计时: 负载不均与同步开销', fontsize=12, fontweight='bold', pad=8)
ax4.axis('off')

# 模拟负载不均
comp = [2.8, 2.2, 1.5, 2.0]  # 计算时间比例
bar_h = 0.5
base_y = 3.8

# 横轴范围
max_c = max(comp)
sync_part = 0.6

for g in range(4):
    y = base_y - g * 0.7
    bw = comp[g]

    # 计算阶段
    rect = mpatches.FancyBboxPatch((0, y), bw, bar_h,
                                    boxstyle="round,pad=0.04",
                                    facecolor=gc[g], alpha=0.7, ec='white')
    ax4.add_patch(rect)
    ax4.text(bw/2, y+bar_h/2, f'Core {g} 计算', ha='center', va='center',
             fontsize=7, color='white', fontweight='bold')
    ax4.text(bw+0.1, y+bar_h/2, f'{comp[g]:.1f}us', va='center', fontsize=6, color=gc[g])

    # barrier 等待 (非最慢核)
    if comp[g] != max_c:
        w = max_c - comp[g]
        rect_w = mpatches.FancyBboxPatch((bw, y), w, bar_h,
                                          boxstyle="round,pad=0.04",
                                          facecolor='#DDD', alpha=0.5, ec='#BBB', hatch='///')
        ax4.add_patch(rect_w)
        ax4.text(bw+w/2, y+bar_h/2, '等待 barrier', ha='center', va='center',
                 fontsize=6, color='#888')

    # 时间标注
    ax4.text(0, y-0.15, f'min={min(comp):.1f}  max={max_c:.1f}  ratio={max_c/min(comp):.2f}x',
             fontsize=6, color='#E74C3C', fontweight='bold')

# t0, t1, t2 标注
ax4.axhline(y=base_y - 4*0.7 - 0.1, xmin=0, xmax=1.2, color='#999', lw=1)
ax4.text(0, base_y - 4*0.7 - 0.35, 't0: 开始', fontsize=6, color='#666')
ax4.text(max_c-0.5, base_y - 4*0.7 - 0.35, 't1: 最快核到达 barrier', fontsize=6, color='#666')
ax4.text(max_c, base_y - 4*0.7 - 0.35, 't2: 最慢核到达', fontsize=6, color='#666',
         ha='right')

# sync 时间块
sync_y = base_y - 4*0.7 - 0.1
rect_s = mpatches.FancyBboxPatch((max_c, sync_y - 0.5), sync_part, 0.4,
                                  boxstyle="round,pad=0.04",
                                  facecolor='#F5B042', alpha=0.5, ec='#F18F01')
ax4.add_patch(rect_s)
ax4.text(max_c + sync_part/2, sync_y - 0.3, f'sync:\nbarrier 等待',
         ha='center', fontsize=6, color='#C77A00', fontweight='bold')
ax4.text(max_c + sync_part/2, sync_y - 0.75, f'total = max(compute) + sync',
         ha='center', fontsize=7, color='#C77A00')

# 公式框
ax4.text(0.5, -1.0, '每个核独立计时:\ncore_times[g] = (t1 - t0)\n\nsync = master 核 barrier 等待时间\n\n总耗时 = max(core_times) + sync',
         fontsize=7, color='#444', va='top',
         bbox=dict(boxstyle='round', facecolor='#F5F5F5', ec='#CCC'))

ax4.set_xlim(-0.5, max_c + sync_part + 0.5)
ax4.set_ylim(-1.5, 4.5)

plt.tight_layout(rect=[0, 0, 1, 0.95])
out_path = '/home/cuihaitao/cache_test/exp6_task_assignment.png'

# 修复字体缺失问题：先存为 PDF 再用 Pillow 转 PNG
plt.savefig('/home/cuihaitao/cache_test/exp6_task_assignment.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'可视化已保存: {out_path}')
