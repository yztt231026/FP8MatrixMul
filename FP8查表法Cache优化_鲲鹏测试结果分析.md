# 鲲鹏服务器 FP8 查表法测试结果分析

## 1. 测试环境确认

### 1.1 硬件参数

| 项目 | 确认值 |
|------|--------|
| CPU | HiSilicon Kunpeng, 4路 |
| 物理核 | 320 (80核/socket × 4) |
| 逻辑核 | 640 (SMT 2:1) |
| SVE 宽度 | **256-bit** (svcntw()=8, svcntb()=32) |
| L1d | 64 KiB/核 |
| L2 | 1280 KiB/核 |
| L3 | 70 MiB/NUMA node |
| NUMA | 8 nodes, 各 80 CPUs |

> SVE 宽度为 **256-bit**（鲲鹏 920 的一般是 128-bit，这颗芯片是增强版本），意味着单次 gather 能加载 8 个 float。

### 1.2 运行限制

测试输出显示 `NUMA Node 0 CPUs: 1 个`，且所有并行测试未能有效多核扩展，说明 **进程被绑核到 1 个 CPU 上**。可能原因：

- 运行环境（容器/SLURM/SSH）设置了 CPU 亲和性
- `taskset` 限制了可用 CPU
- `cgroup` 限制了 CPU 配额

> **后续所有多核测试需要解除绑核**：`numactl --cpunodebind=0 --membind=0 ./fp8_server`

---

## 2. 实验结果分析

### 2.1 实验 1：SVE 向量化加速比

```
Version             Time(us)        GOP/s         Speedup
Scalar (标量)      94370.2          1.42          1.00×
SVE gather         88041.8          1.52          1.07×
```

**结论：SVE gather 仅比标量快 7%。**

#### 原因分析

SVE gather 指令 `svld1_gather_u32index_f32` 虽然一次加载 8 个 float，但查表访存是**随机访问**而非连续访问：

```
标量版本（8 次加载）：
  load table[addr0]  → L2 hit
  load table[addr1]  → L2 hit
  load table[addr2]  → L2 hit
  ...  共 8 次独立加载，out-of-order 执行可重叠

SVE gather 版本（1 条指令）：
  gather {addr0..addr7} → 8 路 L2 hit → 串行化到 1 条结果
```

关键因素：

1. **瓶颈在 L2 延迟，不在计算**：LUT(256K) > L1d(64K)，每次查表必然 L2 miss + L2 hit。L2 延迟 ~12 周期，8 次 gather 共需 ~96 周期，而 8 次标量加载在大乱序执行下也可重叠到 ~40-60 周期。

2. **鲲鹏 SVE gather 微架构**：鲲鹏的 gather 指令微码化了多个 μop，每个元素独立访问 L2，与标量加载的延迟差异不大。Gather 的真正优势在**连续可预取模式**（如 AoS → SoA），而非随机查表。

3. **编译器优化**：标量版本的 `(idxA << 8) | idxB` 被编译器优化为地址计算 + 加载，循环展开后 ILP（指令级并行）充足。

#### 推论

对于查表法这种**随机 gather + memory-bound** 场景，SVE 向量化不是提升性能的有效手段。**计算密集型**的 INT8 I8MM (`svmmla`) 才能充分发挥 SVE 优势。

---

### 2.2 实验 2：Tile 尺寸扫描

```
实验2标题: Tile 尺寸扫描 (SVE + 1 线程)
```

由于进程被绑核，实际运行在 **1 线程** 下。结果如下：

| Ni | Sj | 工作集 | GOP/s | 分析 |
|-----|-----|--------|-------|------|
| 1-4 | 任意 | 260-522K | 1.44-1.58 | 小 tile，外循环开销大 |
| 8-16 | 任意 | 264-552K | 1.64-1.74 | 中等 tile，开销均衡 |
| 32-128 | 任意 | 277-832K | 1.67-1.74 | 大 tile，开销最小 |

**最佳：Ni=16, Sj=512 → 1.74 GOP/s**

#### 关键发现

1. **所有 tile 都在 L2 内**（最大 832K < 1280K L2），所以没有 cache miss 差异。

2. 性能差异来自**单线程下 tile 循环的开销**：
   - Ni=1: `collapse(2)` 产生 `256×512 = 131072` 个 tile → 外层循环开销大
   - Ni=128: 只产生 `2×512 = 1024` 个 tile → 开销小

3. **1 线程下 tile 性能差异仅 ~20%**，对比 x86 的 10 线程下 2.13→18.02 GOP/s（~8× 差异），说明**没有并行化是最大瓶颈**。

#### 修正预期

如果正确绑定 80 核，预期：

| 线程数 | 理论加速比 | 预期 GOP/s |
|--------|-----------|-----------|
| 1 | 1× | ~1.7 |
| 80 (无 tile) | ~10-20× | ~17-34 |
| 80 (有 tile 32×128) | ~15-30× | ~25-50 |

实际受内存带宽限制，查表法的加速比通常低于计算密集型任务。

---

### 2.3 实验 3：单 NUMA Node 内多核扩展

```
NUMA Node 0 CPUs: 1 个     ← 绑核问题！

Threads   Time(us)      Speedup     Efficiency  GOP/s
1         78927.3       1.00×      100%        1.70
```

只有 1 线程被测试。根因：

```cpp
// matmul_fp8_server.cpp 第 641 行
for (int i = first_cpu; i < first_cpu + 80 && i < omp_get_max_threads(); i++)
    cpus.push_back(i);
```

`omp_get_max_threads()` 返回当前进程可用的 CPU 数。如果绑核到 1 个 CPU，则返回 1 → 循环只添加 1 个 CPU。

**修复方法：** 使用 `numactl` 或解除 `taskset` 限制。

---

### 2.4 实验 4：跨 NUMA 扩展

```
Threads       Time(us)      GOP/s       Speedup
1             313179.6      1.71        1.00×
```

同样只有 1 线程被测试，原因同上。

---

### 2.5 实验 5：矩阵规模扩展 × Tile 优化

这是问题表现最明显的实验。80 线程确实创建了（因为 `omp_set_num_threads(80)` 被调用），但运行在 1 个 CPU 上：

```
Scale   Threads   Tiling    Time(us)      GOP/s      对比 1 线程
1×      1         none      77051.8       1.74       —
1×      80        none      96397.3       1.39       × 更慢！
1×      80        32×128    96268.8       1.39       × 更慢！

2×      1         none      308946.3      1.74       —
2×      80        none      334940.9      1.60       × 更慢！
2×      80        32×128    333703.3      1.61       × 更慢！
```

**80 线程比 1 线程慢 13-25%！**

原因：80 个 OpenMP 线程在 1 个 CPU 上**时间片轮转**：
- 线程切换开销（每次切换 ~1-10 μs）
- **L2 cache 污染**：每个线程的时间片内将 LUT 加载到 L2，切换后立即被下一个线程的 LUT 访问驱逐
- 实际计算时间远小于线程切换开销

这也说明 **tile 优化在没有真正并行的情况下无法发挥作用**。

#### 整数溢出 Bug

4× 和 8× 的 GOP/s 计算因 `int` 溢出错误：

```
实验值: 4× GOP/s = -1.68  (应为 ~1.74)
实验值: 8× GOP/s = 0.00   (应为 ~1.74)
```

修复后：`calc_gops` 中的 `N * S * L` 改为 `double(N) * double(S) * double(L)`。

---

## 3. 核心技术结论

### 3.1 SVE gather 不适合随机查表

| 访问模式 | SVE gather 加速比 | 原因 |
|---------|-----------------|------|
| 连续（AoS→SoA） | ~4-8× | 硬件预取 + 连续带宽 |
| **随机（查表）** | **~1.07×** | L2 延迟瓶颈，串行化 |

**建议：** 查表法的核心瓶颈是访存模式，不是计算吞吐量。向量化不能有效解决这个问题。

### 3.2 查表法本质是 memory-bound

所有规模下单线程稳定在 ~1.7 GOP/s，印证了查表法的 **memory-bound** 特性。提升路径：

1. **减少 LUT 大小** → 放进 L1d（64K）：
   - `float` → `__fp16`: 256K → 128K（仍 > 64K ❌）
   - 二级查表：先用粗表定位，再精调（可实现 L1 驻留 ✅）

2. **硬件原生 FP8**：如果 CPU 支持原生 FP8 计算，可完全绕过查表

3. **增加内存带宽**：多通道 DDR + 绑核到同一 NUMA node

### 3.3 此平台的最大优化空间

本服务器的真正优势在 **320 核 × 256-bit SVE**，但内存带宽是共享瓶颈。建议：

| 方向 | 预期收益 | 关键点 |
|------|---------|--------|
| INT8 I8MM (`svmmla`) | ~10-100× | 计算密集，充分发挥 SVE + 多核 |
| 查表 + 正确多核绑定 | ~5-15× | 受内带宽限制，仍远弱于 I8MM |
| 查表 + LUT 缩小 | ~2-4× | L1 命中 vs L2 命中 |

---

## 4. 正确运行方式

### 4.1 确认当前 CPU 亲和性

```bash
# 检查当前进程可用 CPU
taskset -p $$
numactl --show

# 检查系统总核数
lscpu | grep "CPU(s)"
nproc
```

### 4.2 重新运行测试

```bash
# 方式1：绑定到 NUMA node 0 的所有核
numactl --cpunodebind=0 --membind=0 \
  OMP_NUM_THREADS=80 OMP_PLACES=cores OMP_PROC_BIND=close \
  ./fp8_server

# 方式2：使用所有核（4路全开）
OMP_NUM_THREADS=320 OMP_PLACES=cores OMP_PROC_BIND=spread \
  ./fp8_server
```

### 4.3 对比查表 vs I8MM

如果服务器支持 SVE I8MM（flags 中有 `svei8mm`），可显著对比：

```bash
# 利用已有的 matmul_int8_i8mm_complete 函数
# 对比 FP8 查表 (1.7 GOP/s) vs INT8 I8MM (预计 50-200 GOP/s)
```

---

## 5. 修正后代码

已修复的问题：

| 问题 | 修复 |
|------|------|
| `calc_gops` 整数溢出 | `double(N) * double(S) * double(L)` |
| 绑核导致无法多核 | 需用户侧解除限制，代码中添加绑核提示 |
| NUMA CPU 列表错误 | 代码中 `first_cpu_on_node` 逻辑已添加诊断输出 |

重新拉取最新代码：

```bash
git pull
# 确认可用核数后再运行
taskset -p $$ | grep -oP 'mask: \K.*' | perl -e 'print "当前可用 CPU 数: " . (split(//, `taskset -p $$`)[-1] =~ tr/,//+1) . "\n"'
numactl --cpunodebind=0 --membind=0 OMP_NUM_THREADS=80 OMP_PLACES=cores OMP_PROC_BIND=close ./fp8_server
```
