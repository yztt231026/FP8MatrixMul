# FP8 查表法矩阵乘法 — 鲲鹏服务器 Cache 优化实验设计

## 1. 平台分析

### 1.1 硬件参数

| 项目 | 规格 |
|------|------|
| CPU | HiSilicon Kunpeng (鲲鹏) |
| 物理核 | 320 (4路 × 80核/路) |
| 逻辑核 | 640 (SMT 2:1) |
| NUMA | 8 nodes (每 socket 分 2 个 NUMA domain) |
| **L1d** | **64 KiB/核** × 320 = 20 MiB |
| **L2** | **1.25 MiB/核** × 320 = 400 MiB |
| **L3** | **70 MiB/NUMA node** × 8 = 560 MiB |
| 指令集 | ARM SVE (128-bit), SVE I8MM, FP16 |

### 1.2 关键分析

LUT 大小 = 256 KiB，和 x86 平台一样处于 L1 和 L2 之间：

| 层级 | 鲲鹏 | LUT 能否放下 |
|------|------|-------------|
| L1d | 64 KiB (比 x86 的 48K 大) | ❌ 256K > 64K |
| L2 | 1280 KiB (和 x86 相同) | ✅ 256K < 1280K |
| L3/Node | 70 MiB (远大于 x86 的 25M) | ✅ |

**核心差异：**
- L3 巨大 (70M/node) → 需要更大的矩阵才能撑爆 L3
- NUMA 8 节点 → 跨 NUMA 访存延迟是关键瓶颈
- SVE 向量指令 → 查表 gather 可向量化

---

## 2. 实验设计

### 2.1 实验 1：SVE 向量化加速比

**目的：** 验证 SVE gather 指令相比标量查表的单核加速比

**条件：**
- 矩阵 256×1024 (工作集 ~896 KiB，全部在 L2 内)
- 1 线程，对比 `lookup_scalar` vs `lookup_sve`

**预期：** SVE gather 一次加载 `svcntw()` 个 float（鲲鹏 128-bit SVE = 4 个），理论上限 4×，实际受 L2 gather 延迟限制，预计 **1.5~3×**。

### 2.2 实验 2：Tile 尺寸扫描

**目的：** 找到鲲鹏服务器上的最优 tile 尺寸

**条件：**
- 矩阵 256×1024
- SVE + OpenMP，**80 线程（1 个 NUMA node）**
- Ni ∈ {1,2,4,8,16,32,64,128}, Sj ∈ {8,16,32,64,128,256,512}

**Tile 工作集预估：**

| Ni | Sj | 工作集 | Cache |
|-----|-----|--------|-------|
| 32 | 128 | 352 KiB | **L2** |
| 64 | 256 | 608 KiB | **L2** |
| 128 | 512 | 1.06 MiB | **L2** (略小于 1280K) |
| 256 | 1024 | 2.25 MiB | L3 |

**预期：** L2 内的 tile 性能优于 L3 的 tile。最优组合可能在 Ni=16~64, Sj=64~256。

### 2.3 实验 3：单 NUMA Node 内多核扩展

**目的：** 测试 1 个 NUMA node 内 (80 核) 的并行扩展性

**条件：**
- SVE 无 tile（方便对比 baseline）
- 线程数 1→2→4→8→16→32→64→80
- **线程绑定到同一个 NUMA node 的 CPU**（避免跨 NUMA 影响）

**预期：**
- 1→16 核：良好扩展（~12-16×）
- 16→80 核：开始受内存带宽限制，效率下降
- 查表法是 memory-bound，80 核加速比预计 ~20-40×

### 2.4 实验 4：跨 NUMA 扩展

**目的：** 测试跨 8 个 NUMA node 的极限扩展性

**条件：**
- 矩阵 512×2048
- SVE + Tiling (32×128)
- 线程数 1, 80, 160, 240, 320, 640
- OpenMP `OMP_PROC_BIND=spread`（跨 NUMA 散布）
- 对比 tile 版本和无 tile 版本

**NUMA 拓扑：**

```
Socket 0        Socket 1        Socket 2        Socket 3
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ Node0    │   │ Node2    │   │ Node4    │   │ Node6    │
│ 0-39     │   │ 80-119   │   │ 160-199  │   │ 240-279  │
├──────────┤   ├──────────┤   ├──────────┤   ├──────────┤
│ Node1    │   │ Node3    │   │ Node5    │   │ Node7    │
│ 40-79    │   │ 120-159  │   │ 200-239  │   │ 280-319  │
└──────────┘   └──────────┘   └──────────┘   └──────────┘
   L3 70M         L3 70M         L3 70M         L3 70M
```

**预期：**
- 1→80 (单 node)：接近线性
- 80→160 (跨 2 node)：效率下降，跨 NUMA 延迟 ~1.3-1.5×
- 160→320 (全核)：效率进一步下降，内存带宽竞争
- 320→640 (SMT 超线程)：收益很小 (< 20%)

### 2.5 实验 5：矩阵规模扩展

**目的：** 验证不同矩阵规模下 tile 优化的效果

**条件：**

| Scale | N | S | 工作集 | 预计 cache |
|-------|---|---|--------|-----------|
| 1× | 256 | 1024 | ~896 KiB | L2/核 |
| 2× | 512 | 2048 | ~3.5 MiB | L3/Node |
| 4× | 1024 | 4096 | ~14 MiB | L3/Node |
| 8× | 2048 | 8192 | ~56 MiB | L3/Node |

对比三种配置：
- 1 线程（基线）
- 80 线程，无 tile
- 80 线程，tile 32×128

**关键假设：** 鲲鹏 L3 = 70 MiB/node，所以 8× 规模(56M)仍在 L3 内。要看到 DRAM 效果需要 16× 以上。但本实验重点展示 **tile 的 L2 驻留优势**——即使 L3 能放下，L2 的更低延迟仍能带来收益。

---

## 3. 编译与运行

### 3.1 编译

```bash
# 依赖: libnuma-dev
sudo yum install -y numactl-devel   # CentOS/openEuler
# 或
sudo apt install -y libnuma-dev      # Ubuntu

# 编译
g++ -O3 -fopenmp -march=armv8.2-a+sve \
    -o fp8_server matmul_fp8_server.cpp \
    -std=c++17 -lnuma
```

### 3.2 运行

```bash
# 默认运行全部实验
./fp8_server

# 精细控制（推荐）
OMP_PLACES=cores OMP_PROC_BIND=close ./fp8_server

# 绑定到特定 NUMA node
numactl --cpunodebind=0 --membind=0 ./fp8_server
```

### 3.3 Cache Miss 分析

```bash
# 安装 perf (openEuler)
sudo yum install -y perf

# 对比无 tile / 有 tile 时的 L2 miss
perf stat -e L1-dcache-load-misses,l2d_cache_refill,ll_cache_miss \
    OMP_NUM_THREADS=80 ./fp8_server
```

---

## 4. 预期结果

### 4.1 定性预测

| 实验 | 关键结论 |
|------|---------|
| SVE vs Scalar | SVE gather ~2-4×，但受限 L2 延迟 |
| Tile 扫描 | 最优 tile 工作集在 300-600 KiB (L2) |
| 单 NUMA 扩展 | 80 核加速比 ~20-40× (查表法 memory-bound) |
| 跨 NUMA | 跨 node 访存开销显著，spread binding 优于 close |
| 矩阵规模 | Tile 在所有规模下都有收益，大矩阵收益更明显 |

### 4.2 与 x86 结果对比

| 项目 | x86 i7-12700 | ARM 鲲鹏 | 差异原因 |
|------|-------------|---------|---------|
| L1d | 48 KiB | 64 KiB | 鲲鹏略大 |
| L2 | 1280 KiB | 1280 KiB | 相同 |
| 核数 | 10 | 320 | 32× 更多核 |
| NUMA | 1 node | 8 nodes | 鲲鹏需要 NUMA-aware 编程 |
| SVE | 不支持 | 128-bit | 可向量化查表 |
| 最优 tile | 2×8 ~ 4×16 | 预计 16×64 ~ 32×128 | 鲲鹏 L1 更大 |

### 4.3 查表法 vs 直接计算

查表法的优点是**绕过 FP8 的硬件不支持问题**，缺点是：

- **Memory-bound**：每次查表需要一次随机 gather 访存
- **加速比受限**：远低于计算密集型的 INT8 I8MM（`svmmla`）
- **跨 NUMA 敏感**：gather 访存的延迟对 NUMA 距离更敏感

如果目标平台的 SVE 支持 `svmmla`（I8MM），建议也一并对比测试：

```bash
# 对比查表法和 I8MM 矩阵乘
# 用户已有 matmul_int8_i8mm_complete() 函数
```

---

## 5. 代码结构

```
matmul_fp8_server.cpp
├── 数据生成
├── 计算核心
│   ├── lookup_scalar()          — 标量（串行参考）
│   ├── lookup_sve()             — SVE gather（串行）
│   ├── lookup_sve_omp()         — SVE + OpenMP
│   └── lookup_sve_tiled_omp()   — SVE + 分块 + OpenMP
├── NUMA 工具
│   ├── print_numa_info()
│   ├── first_cpu_on_node()
│   └── pin_to_cpu_list()
├── 实验函数
│   ├── exp1_sve_speedup()       — SVE vs 标量
│   ├── exp2_tile_sweep()        — Tile 扫描 (80核)
│   ├── exp3_numa_scaling()      — 单 NUMA node 扩展
│   ├── exp4_cross_numa()        — 跨 NUMA 扩展
│   └── exp5_matrix_scaling()    — 矩阵规模扩展
└── main()
```
