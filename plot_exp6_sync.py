#!/usr/bin/env python3
"""解释同步时间先升后降的原因"""

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

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle('同步时间先升后降的原因分析', fontsize=15, y=1.05)

G = np.array([1, 2, 4, 8, 16, 32, 64])
sync = np.array([1.6, 14.4, 1073.1, 2803.6, 2573.6, 460.3, 602.6])
# 每核平均元素数 = 512 / G
avg_elems = 512 / G
# 预计 atomic 操作总次数 = G * avg_elems = 512 (常数，与G无关)
# 但 atomic 竞争激烈程度与 G 正相关（更多核争同一缓存行）
# 负载不均衡 = 预计不均衡度（二项分布 CV ≈ 1/sqrt(avg_elems)）
cv = 1 / np.sqrt(avg_elems)  # 变异系数
# atomic 竞争激烈程度：正比于 G（核数），但每核操作次数反比于 G
# 总 atomic 操作次数 = N*S*(512) = 常数
# 但每次 atomic 的延迟与参与核数成正比（MESI 传递）
# 所以 atomic 总时间 ≈ G * (512/G) = 常数？不对...
# 实际上 atomic 竞争是 super-linear 的——更多核争抢同一缓存行，每次 atomic 延迟更高

# 更准确的分析：
# atomic 总延迟 = (每核 atomic 次数) * (每次 atomic 的延迟)
# 每核 atomic 次数 = N*S*(512/G) ∝ 1/G
# 每次 atomic 的延迟 ∝ G（更多核争抢，MESI 传递更慢）
# 所以 atomic 总延迟 ∝ (1/G) * G = 常数？但实测不是常数

# 关键在于：
# 1. G<8: atomic 竞争主导，每次 atomic 的延迟随 G 超线性增长
# 2. G>=16: 每核元素数变少，atomic 总操作次数=常数，但 G 增大使每次 atomic 延迟继续增大
# 3. G=32: 奇怪的是 sync 反而大幅下降

# 实际上 sync 时间 = barrier 等待时间 ≠ atomic 总开销
# barrier 等待 = 最慢核 - 最快核
# 当每核元素数很少时（G=32: ~16个, G=64: ~8个），绝对时间差变小
# 即使不均衡度大，但每核总量小，max - min 的绝对值也小

# 所以正确解释：
# G=8~16: 负载不均大 + atomic 竞争大 → sync 最大
# G=32~64: 每核元素很少，max-min 绝对差值变小 → sync 变小

# 图1: 同步时间 vs G
ax = axes[0]
ax.plot(G, sync, 'o-', color='#E74C3C', linewidth=2, markersize=10)
ax.set_xlabel('G (子表数)', fontsize=11)
ax.set_ylabel('同步时间 (μs)', fontsize=11)
ax.set_title('同步时间随 G 的变化', fontsize=12)
ax.set_xticks(G)
for i, v in enumerate(sync):
    ax.text(G[i], v + 150, f'{v:.0f}', ha='center', fontsize=9, fontweight='bold')
# 分阶段标注
ax.axvspan(0.5, 3.5, alpha=0.05, color='#2ECC71')
ax.text(2, 200, 'atomic 竞争\n上升期', ha='center', fontsize=8, color='#2ECC71', fontweight='bold')
ax.axvspan(3.5, 16.5, alpha=0.05, color='#F39C12')
ax.text(10, 200, '竞争峰值\nG=8~16', ha='center', fontsize=8, color='#F39C12', fontweight='bold')
ax.axvspan(16.5, 64.5, alpha=0.05, color='#3498DB')
ax.text(40, 200, '每核元素少\n绝对差变小', ha='center', fontsize=8, color='#3498DB', fontweight='bold')

# 图2: 负载不均与每核元素数
ax = axes[1]
ax2 = ax.twinx()

line1 = ax.plot(G, avg_elems, 's-', color='#3498DB', linewidth=2, markersize=8, label='每核平均元素数')
line2 = ax2.plot(G[1:], cv[1:], 'D--', color='#E74C3C', linewidth=2, markersize=8, label='不均衡度 (CV)')

ax.set_xlabel('G (子表数)', fontsize=11)
ax.set_ylabel('每核平均元素数', fontsize=11, color='#3498DB')
ax2.set_ylabel('不均衡度 CV', fontsize=11, color='#E74C3C')
ax.set_title('每核元素数 vs 不均衡度', fontsize=12)
ax.set_xticks(G)
ax.set_xscale('log', base=2)

lines = line1 + line2
ax.legend(lines, [l.get_label() for l in lines], fontsize=9, loc='center')
ax.tick_params(axis='y', labelcolor='#3498DB')
ax2.tick_params(axis='y', labelcolor='#E74C3C')

# 图3: 两个因素的交互说明
ax = axes[2]
ax.axis('off')

text = """
同步时间 = barrier 等待 ≈ max(compute) - min(compute)

两个因素相互抵消：

G 增大 → 每核元素数减少（512/G）
         → 每核 atomic 操作次数减少 ✓
         → max-min 的绝对值减小 ✓

G 增大 → 核间执行时间差异相对增大 ×
         （不均衡度 CV ~ 1/√每核元素数）

在 G=8~16 时两个因素达到最差平衡：
  · atomic 竞争已显著（8~16核争一条缓存行）
  · 每核元素还有 32~64 个，绝对差仍大

到 G=32~64 时：
  · 每核仅 16~8 个元素
  · 即使不均衡度 2×，max-min 绝对值也小
  · atomic 操作频率低（每次循环仅几次）
  → 同步时间大幅下降
"""

ax.text(0.05, 0.95, text, fontsize=9, color='#333', va='top',
        transform=ax.transAxes, fontfamily='monospace')

plt.tight_layout()
out_path = '/home/cuihaitao/cache_test/exp6_sync_analysis.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'可视化已保存: {out_path}')
