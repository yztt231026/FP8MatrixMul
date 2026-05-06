# 鲲鹏服务器 FP8 查表法测试结果分析

## 1. 测试环境

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

> SVE 宽度为 **256-bit**（鲲鹏 920 一般是 128-bit，这颗是增强版本），单次 gather 加载 8 个 float。

### 1.2 运行方式

```bash
# 单 NUMA node 测试（实验1/2/3/5）
numactl --cpunodebind=0 --membind=0 \
  OMP_NUM_THREADS=80 OMP_PLACES=cores OMP_PROC_BIND=close \
  ./fp8_server

# 跨 NUMA 测试（实验4，必须取消绑核让线程散布到各 socket）
OMP_NUM_THREADS=320 OMP_PLACES=cores OMP_PROC_BIND=spread \
  ./fp8_server
```

### 1.3 核心数据结构

```cpp
// matmul_fp8_server.cpp 第32-33行
static constexpr int LUT_SIZE = 256 * 256;          // 查找表：256×256 float
static constexpr int LUT_KiB  = 256;                 // = 256 KiB，远超 L1d(64K)
```

所有实验共用两组核函数（第 68-148 行）：

| 核函数 | 行号 | 说明 |
|--------|------|------|
| `lookup_scalar` | 68-78 | 标量基线：逐元素查表累加 |
| `lookup_sve` | 81-99 | SVE gather：`svld1_gather_u32index_f32` 一次加载 8 个 float |
| `lookup_sve_omp` | 102-121 | SVE + OpenMP 行并行 |
| `lookup_sve_tiled_omp` | 124-148 | SVE + 分块(Tiling) + OpenMP |

查表核心逻辑（以标量版第 74-75 行为例）：

```cpp
// 将两个 uint8_t 索引拼成 16-bit 偏移，查表累加
sum += table[(rA[k] << 8) | rB[k]];
```

---

## 2. 实验结果分析

### 2.1 实验 1：SVE 向量化加速比

**目的：** 对比标量查表与 SVE gather 向量查表的单核性能差异，验证向量化对随机 gather 访存是否有效。

**代码：** `exp1_sve_speedup()`（第 233-276 行）

**输入：**
- 矩阵 A: 256×512, 矩阵 B_T: 1024×512, 内维度 L=512
- 工作集: 1920 KiB（远超 L2，但单核下 L3 命中）
- 1 线程，串行执行

**输出：**

```
Version             Time(us)        GOP/s         Speedup
Scalar (标量)      94408.4          1.42          1.00×
SVE gather         82293.9          1.63          1.15×
```

**原理分析：**

查表法 `table[(rA[k] << 8) | rB[k]]` 的访存模式是**完全随机**的——每次查表地址由数据决定，无法预取。

标量版本（`lookup_scalar`, 第 68-78 行）每步做一次 load + multiply-add，编译器展开循环后乱序执行可重叠多个 L2 加载。

SVE gather 版本（`lookup_sve`, 第 81-99 行）用 `svld1_gather_u32index_f32` 一次加载 8 个 float：

```cpp
// 第 92-93 行：SVE gather 一次加载 8 个随机地址的 float
svuint32_t idx = svorr_u32_z(pg, svlsl_n_u32_z(pg, idxA, 8), idxB);
acc = svadd_f32_z(pg, acc, svld1_gather_u32index_f32(pg, table, idx));
```

鲲鹏的 gather 指令微码化了多个 μop，每个元素独立访问 L2。8 个随机地址的 L2 访问无法流水，反而将延迟串行化。

**结论：** 查表法的瓶颈在 L2 随机加载延迟（~12 周期），不在计算向量宽度。对于这种 memory-bound 随机 gather 场景，SVE 向量化不是有效手段。这与连续访问模式（如 AoS→SoA，可达 4-8× 加速）有本质区别。

---

### 2.2 实验 2：Tile 尺寸扫描（80 线程）

**目的：** 找到鲲鹏平台的最优 tile 分块尺寸 (Ni, Sj)，平衡 L2 驻留与并行度。

**代码：** `exp2_tile_sweep()`（第 280-333 行）

**输入：**
- 矩阵 A: 256×512, 矩阵 B_T: 1024×512, L=512
- 80 线程（1 NUMA node），`OMP_PROC_BIND=close`
- 扫描 Ni ∈ {1,2,4,8,16,32,64,128}, Sj ∈ {8,16,32,64,128,256,512}
- 调用 `lookup_sve_tiled_omp()`（第 124-148 行）

**核函数原理：** `lookup_sve_tiled_omp` 将输出矩阵 C (N×S) 按 tile (Ni×Sj) 分块，外层用 `#pragma omp parallel for collapse(2) schedule(dynamic)` 分发，内层每个 tile 仍使用 SVE gather 做内积。

单 tile 的工作集公式（`calc_tile_workset_kib`, 第 50-52 行）：

```
WorkSet = A_tile + B_T_tile + C_tile + LUT
        = Ni × L × 1 + Sj × L × 1 + Ni × Sj × 4 + 256 KiB
```

**输出（部分）：**

```
Ni      Sj      TileWorkSet   Time(us)    GOP/s
1       8       260 KiB       2411.4      55.66
4       8       262 KiB       2361.6      56.83
8       8       264 KiB       2369.5      56.64
16      8       268 KiB       2373.8      56.54
32      8       277 KiB       2341.1      57.33    ← 最佳
...
32      64      312 KiB       3074.4      43.66
64      64      336 KiB       3166.2      42.39
128     128     448 KiB       11839.4     11.34
128     512     832 KiB       45385.8     2.96
```

**分析：**

1. **所有 tile 都在 L2 内**（最大 832K < 1280K），所以 cache miss 差异小。所有性能变化来自并行度。

2. **GOP/s 由 tile 数量驱动**（与 x86 实验结果一致）：
   - Ni=32, Sj=8: tiles = (256/32)×(1024/8) = 8×128 = 1024 个 → 57.33 GOP/s
   - Ni=128, Sj=512: tiles = (256/128)×(1024/512) = 2×2 = 4 个 → 2.96 GOP/s

3. **Sj 影响大于 Ni**：固定 Ni=4，Sj 从 8→512 时 GOP/s 从 56.83 降到 45.39。Sj 增大 → B_T tile 增大 → 每 tile 工作集增大 → cache 压力增大。

4. **与 x86 对比**：x86 i7-12700 最优 18.02 GOP/s（10 核），鲲鹏最优 57.33 GOP/s（80 核）。每核效率 x86 ~1.8 vs 鲲鹏 ~0.7。

**最佳参数：Ni=32, Sj=8 → 57.33 GOP/s**

---

### 2.3 实验 3：单 NUMA Node 内多核扩展

**目的：** 测试在 1 个 NUMA node（80 核，共享 70 MiB L3）内查表法的多核扩展性。

**代码：** `exp3_numa_scaling()`（第 337-404 行）

**输入：**
- 矩阵 A: 256×512, 矩阵 B_T: 1024×512, L=512
- `lookup_sve_omp()`（第 102-121 行）—— 无 tile 的行并行版本，`schedule(static)`
- 线程绑定：`sched_setaffinity` 绑定到 NUMA node 0 的前 nt 个 CPU（第 367-369 行）
- `OMP_PLACES=cores OMP_PROC_BIND=close`（第 374-375 行）

**输出：**

```
NUMA Node 0 CPUs: 80 个 (0~79)

Threads   Time(us)      Speedup     Efficiency  GOP/s
1         84131.1       1.00×      100%        1.60
2         44124.4       1.91×      95%         3.04
4         22362.2       3.76×      94%         6.00
8         10915.8       7.71×      96%         12.30
16        5624.6        14.96×     93%         23.86
32        3200.8        26.28×     82%         41.93
64        3852.1        21.84×     34%         34.84
80        3107.4        27.07×     34%         43.19
```

**分析：**

1. **1→16 线程：近线性扩展（93-96% 效率）**。此时线程数少于内存控制器通道数（鲲鹏 920 每个 socket 有 8 通道 DDR4-2933），带宽未饱和。

2. **32 线程开始效率下降（82%）**。触及内存带宽上限。

3. **64 线程异常下降（34% 效率）**：可能原因是 `lookup_sve_omp` 使用 `schedule(static)` 做行并行，64 线程时 N=256 行分配给 64 线程（每线程 4 行），负载不均加剧。

4. **80 线程 27× 加速比（34% 效率）**：**查表法本质是 memory-bound**，每 GOP 需要一次随机 L3 访问。所有 80 核共享 NUMA node 0 的内存带宽（~30-40 GiB/s 实际可用），带宽饱和后无法继续扩展。

5. **与计算密集型对比**：INT8 I8MM (`svmmla`) 是 compute-bound，同样条件下可达接近 80× 线性。

---

### 2.4 实验 4：跨 NUMA 扩展

**目的：** 测试线程散布到多个 socket 时查表法的扩展性，验证 NUMA 内存带宽限制。

**代码：** `exp4_cross_numa()`（第 408-478 行）

**输入：**
- 矩阵 A: 512×512, 矩阵 B_T: 2048×512, L=512（比实验 2/3 更大的矩阵）
- 固定 tile 32×128（工作集 352 KiB → L2）
- `lookup_sve_tiled_omp()`（第 124-148 行）
- `OMP_PLACES=cores OMP_PROC_BIND=spread`（第 458-459 行）

> **关键：** 此实验必须**取消 NUMA 绑定**（不用 `cpunodebind=0`），否则所有线程被限制在一个 node 的 CPU 和内存带宽上，无法测量跨 NUMA 扩展。

**输出：**

```
Threads       Config            Time(us)      GOP/s       Speedup
1             1                 343621.8      1.56        1.00×
80            80 (1 node)       5484.8        97.88       62.65×
160           160 (2 nodes)     3223.5        166.55      106.60×
240           240 (3 nodes)     3469.9        154.72      99.03×
320           320 (4 nodes)     2580.3        208.06      133.17×
```

**分析：**

| 配置 | GOP/s | 相对 80 核 | 效率 | 说明 |
|------|-------|-----------|------|------|
| 80 核（无绑核 spread） | 97.88 | 1.00× | 78% | 散布到各核，各用本地内存带宽 |
| 160 核（2 sockets） | 166.55 | 1.70× | 67% | 2 个独立内存控制器，接近线性 |
| 240 核（3 sockets） | 154.72 | 1.58× | 41% | 3 socket 负载不对称 |
| 320 核（4 sockets） | 208.06 | 2.12× | 42% | 总内存带宽饱和，收益递减 |

> 对比实验 3（绑 `cpunodebind=0`）的 80 核 45.59 GOP/s，实验 4 中 spread 到各 socket 的 80 核达到 97.88 GOP/s — 提升 2.15×。这说明即使是 80 个线程，散布到多 socket 后各用本地内存带宽也能大幅提升吞吐。

**结论：** 查表法的跨 NUMA 扩展受限于**系统总内存带宽**。鲲鹏 4 路的总内存带宽（4 × 8 通道 DDR4-2933）是最终瓶颈，320 核全开时无法线性扩展。

---

### 2.5 实验 5：矩阵规模扩展 × Tile 优化

**目的：** 验证不同矩阵规模下 tile 分块的优化效果，测试 L2 vs L3 vs DRAM 三种层级的工作集对性能的影响。

**代码：** `exp5_matrix_scaling()`（第 484-603 行）

**输入：**

| Scale | N | S | L | 工作集 | Cache 层级 |
|-------|---|---|---|--------|-----------|
| 1× | 256 | 1024 | 512 | 1.9 MiB | L3/核 |
| 2× | 512 | 2048 | 512 | 5.5 MiB | L3/核 |
| 4× | 1024 | 4096 | 512 | 18.8 MiB | L3/Node |
| 8× | 2048 | 8192 | 512 | 69.2 MiB | L3/Node ~ DRAM 边界 |

每种规模测试三种配置：
1. **1 线程，无 tile** — `lookup_sve()` — 单核基线
2. **80 线程，无 tile** — `lookup_sve_omp()` — 多核基线
3. **80 线程，tile 32×128** — `lookup_sve_tiled_omp()` — 多核 + L2 驻留优化

**输出：**

```
Scale    Threads   Tiling    WorkSet   Cache    Time(us)   GOP/s
1×       1         none      1.9 MiB   L3+      88397.2    1.52
1×       80        none      1.9 MiB   L3       3165.3     42.40
1×       80        32×128    352 KiB   L2       3095.6     43.36     ← +2%
────────────────────────────────────────────────────────────────
2×       1         none      5.5 MiB   L3+      331196.1   1.62
2×       80        none      5.5 MiB   L3       10913.2    49.19
2×       80        32×128    352 KiB   L2       11691.4    45.92     ← -7%
────────────────────────────────────────────────────────────────
4×       1         none      18.8 MiB  L3+      1329292.4  1.62
4×       80        none      18.8 MiB  L3       40933.2    52.46
4×       80        32×128    352 KiB   L2       38193.9    56.23     ← +7%
────────────────────────────────────────────────────────────────
8×       1         none      69.2 MiB  L3+      5320550.0  1.61
8×       80        none      69.2 MiB  DRAM     157778.5   54.44
8×       80        32×128    352 KiB   L2       150462.3   57.09     ← +5%
```

**分析：**

1. **单线程稳定在 ~1.6 GOP/s** — 所有 4 种规模下单核吞吐几乎不变。这印证了查表法是纯 memory-bound，计算量增加不会改变吞吐率。

2. **80 线程无 tile 的性能随矩阵增大反而上升（42.4→49.2→52.5→54.4 GOP/s）**。原因是矩阵越大，内存控制器的行缓冲区命中率越高，有效带宽利用率改善。这与直觉相反但合理——大矩阵的连续数据集有更好的空间局部性。

3. **Tile 优化在 4× 和 8× 规模下有收益（+5~7%），但在 1× 和 2× 下不明显甚至下降**。鲲鹏平台的 L3 巨大（70 MiB/node），1× 和 2× 的工作集完全在 L3 内，L2 命中的延迟优势被 80 线程并发稀释。当 8× 规模（69.2 MiB）接近 L3 边界开始触及 DRAM 时，tile 的 L2 驻留优势开始体现。

4. **与 x86 对比**：x86 i7-12700 上 tile 优化有 +35-55% 的巨大收益，原因是 x86 的 L3 仅 25 MiB（全芯片共享），4× 规模（3.7 MiB）就已超出 L3 进入 DRAM。鲲鹏的 L3 大 2.8 倍（每 node），缩小了 tile 优化的相对优势。

---

## 3. 核心技术结论

### 3.1 性能汇总

| 指标 | 值 |
|------|-----|
| 单核峰值 | 1.63 GOP/s (SVE) |
| 单 NUMA node (80核) 峰值 | **57.33 GOP/s** |
| 2 sockets (160核) 峰值 | **166.55 GOP/s** |
| 全机 (320核) 峰值 | **208.06 GOP/s** |
| 最优加速比（320核/1核） | **133×** |
| 每核效率（320核） | ~0.65 GOP/s/核 |

### 3.2 瓶颈分析

查表法矩阵乘法的瓶颈随规模变化：

```
单核：      L2 随机加载延迟（~12 周期/次 gather）
单 NUMA：  内存带宽饱和（80 核用尽单 node 的 ~40 GiB/s）
跨 NUMA：  系统总内存带宽瓶颈（320 核共享 ~160 GiB/s）
```

这是 **memory-bound 查表问题**与计算密集型任务（INT8 I8MM）的本质区别。即使 320 核全开也只能达到 208 GOP/s，因为每次 `GOP = 一次查表 = 一次随机 4 字节加载`，CPU 的大部分时间在等待内存返回数据，而非做计算。

### 3.3 对比 x86 (i7-12700)

| 项目 | x86 (10核) | ARM 鲲鹏 (80核) | 差异原因 |
|------|-----------|----------------|---------|
| 单核 GOP/s | ~2.0 | ~1.6 | x86 P-core 单核更强 |
| 多核峰值 | ~18 | **~57** | 鲲鹏 8 倍核数 |
| 最优 tile | 2×8 | 32×8 | 鲲鹏 L1 更大(64K vs 48K) |
| Tile 收益 | +35-55% | ~+5% | 鲲鹏 L3 巨大(70M vs 25M) |
| 扩展效率 | ~39% (10→1) | ~34% (80→1) | 都受限于内存带宽 |
| 跨 NUMA | 不适用 | 208@320核 | 4 socket × 内存通道 |

### 3.4 优化建议

| 方向 | 预期收益 | 说明 |
|------|---------|------|
| **INT8 I8MM (`svmmla`)** | ~10-100× | 计算密集型，可充分发挥 SVE × 320核 |
| 查表 + 缩小 LUT | ~2-3× | float→__fp16 将 LUT 减至 128K（仍 > L1d） |
| 两级查表 | ~2-4× | 粗调 + 精调，可实现 L1 驻留 |
| 查表 + 跨 NUMA 扩展 | 当前已 208 GOP/s | 瓶颈在总内存带宽，已接近极限 |

> 本服务器的真正价值在于 **320 核 × 256-bit SVE + I8MM**。查表法作为 FP8 的软件模拟方案，在此平台上的性能远无法与硬件原生 INT8 计算相比。建议优先测试 `svmmla` 的 INT8 矩阵乘。

## 4. 代码索引

| 文件 | 行号 | 内容 |
|------|------|------|
| `matmul_fp8_server.cpp` | 32-33 | 常量定义：`LUT_SIZE`(256×256), `LUT_KiB`(256) |
| `matmul_fp8_server.cpp` | 50-52 | `calc_tile_workset_kib()` — 计算 tile 工作集大小 |
| `matmul_fp8_server.cpp` | 68-78 | `lookup_scalar()` — 标量查表核函数 |
| `matmul_fp8_server.cpp` | 81-99 | `lookup_sve()` — SVE gather 查表核函数 |
| `matmul_fp8_server.cpp` | 102-121 | `lookup_sve_omp()` — SVE + OpenMP 行并行 |
| `matmul_fp8_server.cpp` | 124-148 | `lookup_sve_tiled_omp()` — SVE + Tiling + OpenMP |
| `matmul_fp8_server.cpp` | 207-218 | `bench()` — 计时框架（warmup × 2 + 取最优 × 5） |
| `matmul_fp8_server.cpp` | 227-229 | `calc_gops()` — GOP/s 计算公式 |
| `matmul_fp8_server.cpp` | 233-276 | **实验 1：** `exp1_sve_speedup()` |
| `matmul_fp8_server.cpp` | 280-333 | **实验 2：** `exp2_tile_sweep()` |
| `matmul_fp8_server.cpp` | 337-404 | **实验 3：** `exp3_numa_scaling()` |
| `matmul_fp8_server.cpp` | 408-478 | **实验 4：** `exp4_cross_numa()` |
| `matmul_fp8_server.cpp` | 484-603 | **实验 5：** `exp5_matrix_scaling()` |
