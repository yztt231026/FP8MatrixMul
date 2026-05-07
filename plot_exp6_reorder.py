#!/usr/bin/env python3
"""实验6：A 矩阵重排前后对应关系可视化"""

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

fig, axes = plt.subplots(3, 1, figsize=(14, 9))
fig.suptitle('A 矩阵重排前后对应关系 (每行 A[i], G=4)', fontsize=16, y=0.99)

c_g0 = '#E74C3C'
c_g1 = '#3498DB'
c_g2 = '#2ECC71'
c_g3 = '#F39C12'
gc = [c_g0, c_g1, c_g2, c_g3]

row_len = 32

# 生成伪随机数据 — 保证 4 组各有分布
rng = np.random.RandomState(123)
a_vals = []
groups = []
for k in range(row_len):
    # 让每组约有 8 个元素，但不完全均匀
    g = rng.randint(0, 4)
    base = g * 64 + rng.randint(0, 63)
    a_vals.append(base)
    groups.append(g)

# ========== 第1行: 原始 A[i] ==========
ax = axes[0]
ax.set_title('原始 A[i] 行 — 每个 k 有随机的 idxA 值', fontsize=11, pad=5)
ax.axis('off')

y0 = 0.5
x_start = 0.3

# 行号标注
ax.text(x_start - 0.3, y0 + 0.15, 'A[i] 行:', fontsize=9, fontweight='bold', va='center')

for k in range(row_len):
    x = x_start + k * 0.38
    g = groups[k]
    rect = mpatches.FancyBboxPatch((x, y0), 0.34, 0.5,
                                    boxstyle="round,pad=0.02",
                                    facecolor=gc[g], alpha=0.65, ec='white')
    ax.add_patch(rect)
    # 显示 idxA 值
    ax.text(x + 0.17, y0 + 0.35, str(a_vals[k]), ha='center', fontsize=5.5, color='white', fontweight='bold')
    # 显示所属组
    ax.text(x + 0.17, y0 + 0.1, f'g={g}', ha='center', fontsize=4.5, color='white', alpha=0.8)
    # 下方标 k 索引
    ax.text(x + 0.17, y0 - 0.12, f'k={k}', ha='center', fontsize=5, color='#888')

ax.set_xlim(x_start - 0.5, x_start + row_len * 0.38 + 0.5)
ax.set_ylim(-0.3, 1.5)

# ========== 第2行: 按组划分（每个元素标出原始 k）==========
ax = axes[1]
ax.set_title('按 idxA 划分到 G=4 组 — 每个元素保留原始 k 索引', fontsize=11, pad=5)
ax.axis('off')

y_groups = 0.0
group_h = 0.6

for g in range(4):
    y = y_groups + (3 - g) * 0.7  # 组 0 在最上方
    # 组标签
    rect = mpatches.FancyBboxPatch((0.2, y), 0.8, group_h,
                                    boxstyle="round,pad=0.05",
                                    facecolor=gc[g], alpha=0.3, ec=gc[g], lw=1.5)
    ax.add_patch(rect)
    ax.text(0.6, y + group_h/2, f'组 {g}\nidxA∈[{g*64}..{(g+1)*64-1}]',
            ha='center', va='center', fontsize=7, color=gc[g], fontweight='bold')

    # 该组的元素
    elems = [(k, a_vals[k]) for k in range(row_len) if groups[k] == g]
    for idx, (k, v) in enumerate(elems):
        x = 1.5 + idx * 0.42
        rect = mpatches.FancyBboxPatch((x, y + 0.05), 0.38, group_h - 0.1,
                                        boxstyle="round,pad=0.02",
                                        facecolor=gc[g], alpha=0.65, ec='white')
        ax.add_patch(rect)
        ax.text(x + 0.19, y + group_h*0.65, f'{v}', ha='center', fontsize=5.5, color='white', fontweight='bold')
        ax.text(x + 0.19, y + group_h*0.25, f'(k={k})', ha='center', fontsize=4.5, color='white', alpha=0.8)

    # 标注：这些元素来自原始行中不同位置
    if g == 0:
        for idx, (k, _) in enumerate(elems):
            x = 1.5 + idx * 0.42
            # 连线到原始行的对应位置
            orig_x = x_start + k * 0.38 + 0.17
            ax.annotate('', xy=(x + 0.19, y + group_h),
                       xytext=(orig_x, y0 + 0.25),
                       arrowprops=dict(arrowstyle='->', lw=0.6, color=gc[g], alpha=0.4, connectionstyle='arc3,rad=0.2'))

ax.set_xlim(0, row_len * 0.42 + 1)
ax.set_ylim(-0.1, 2.9)

# ========== 第3行: A_grouped + k_indices（内存布局）==========
ax = axes[2]
ax.set_title('A_grouped + k_indices 内存布局 — 组内连续，通过 k_indices 回追原始位置', fontsize=11, pad=5)
ax.axis('off')

# A_grouped
y_a = 0.8
ax.text(0.2, y_a + 0.1, 'A_grouped[]:', fontsize=8, fontweight='bold', va='center')
pos = 0
for g in range(4):
    elems = [(k, a_vals[k]) for k in range(row_len) if groups[k] == g]
    for idx, (k, v) in enumerate(elems):
        x = 2.0 + pos * 0.42
        rect = mpatches.FancyBboxPatch((x, y_a), 0.38, 0.5,
                                        boxstyle="round,pad=0.02",
                                        facecolor=gc[g], alpha=0.65, ec='white')
        ax.add_patch(rect)
        ax.text(x + 0.19, y_a + 0.25, str(v), ha='center', fontsize=5.5, color='white', fontweight='bold')
        pos += 1
    # 组分隔
    if g < 3:
        ax.axvline(x=x + 0.45, ymin=0.12, ymax=0.5, color='#999', lw=0.8, ls='--')

# 数组下标标注
for idx in range(pos):
    x = 2.0 + idx * 0.42
    ax.text(x + 0.19, y_a - 0.12, f'[{idx}]', ha='center', fontsize=4.5, color='#888')

# k_indices (下面一行)
y_k = 0.0
ax.text(0.2, y_k + 0.1, 'k_indices[]:', fontsize=8, fontweight='bold', va='center')
pos = 0
for g in range(4):
    elems = [(k, a_vals[k]) for k in range(row_len) if groups[k] == g]
    for idx, (k, v) in enumerate(elems):
        x = 2.0 + pos * 0.42
        rect = mpatches.FancyBboxPatch((x, y_k), 0.38, 0.5,
                                        boxstyle="round,pad=0.02",
                                        facecolor='#E8E8E8', alpha=0.8, ec='#CCC')
        ax.add_patch(rect)
        # 显示原始 k 索引
        ax.text(x + 0.19, y_k + 0.25, f'k={k}', ha='center', fontsize=5.5, color='#555', fontweight='bold')
        pos += 1

# 垂直对应连线（选几个）
for idx in [0, 4, 12, 20]:
    if idx >= pos:
        break
    x = 2.0 + idx * 0.42
    ax.plot([x + 0.19, x + 0.19], [y_a + 0.5, y_k + 0.5],
            lw=0.8, color='#AAA', ls=':')

# 使用方式标注
ax.text(0.3, -0.8, '核 g 通过 row_start[i*G+g] 拿到起始位置，通过 row_count[i*G+g] 拿到元素个数',
        fontsize=8, color='#444')
ax.text(0.3, -1.1, '然后连续读取 A_grouped[start .. start+count-1] 和对应的 k_indices',
        fontsize=8, color='#444')
ax.text(0.3, -1.4, '用 k_indices[t] 回追原始 k → 读 B_T[j][k_indices[t]]',
        fontsize=8, color='#444', fontweight='bold')

ax.set_xlim(0, row_len * 0.42 + 1)
ax.set_ylim(-2.0, 1.8)

plt.tight_layout(rect=[0, 0, 1, 0.95])
out_path = '/home/cuihaitao/cache_test/exp6_reorder_mapping.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'可视化已保存: {out_path}')
