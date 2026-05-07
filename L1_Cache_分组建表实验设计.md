# FP8 查表法 L1 Cache 分组建表实验设计（服务器 SVE 版）

## 1. 核心思路

### 1.1 问题

当前 LUT 256 KiB > 鲲鹏 L1d 64 KiB，每次查表命中 L2（~12 周期）。如果能放进 L1（~4 周期），单次查表延迟降低 ~3×。

### 1.2 方案：切 LUT + 多核协作

将 LUT 按 idxA 范围切为 **G 个子表**（float 存储，直接复用已有 LUT 数据）。G 个核同时计算**同一个 (i,j)**，各自查自己的子表，最后用 `#pragma omp atomic` 规约部分和。

```
原始查表：         table[A[i][k]][B_T[j][k]]   每次查表 → L2 (~12 cyc)
                   单核处理所有 k

G 个子表 + G 核：
  核0: subtable_0[A[i][k]-0][B_T[j][k]]    k∈组0 → 子表 L1 命中 (~4 cyc)
  核1: subtable_1[A[i][k]-off][B_T[j][k]]  k∈组1 → 子表 L1 命中 (~4 cyc)
  ...
  核G-1: ... → 子表 L1 命中 (~4 cyc)

  结果 = 部分和_0 + 部分和_1 + ...  (atomic 累加)
```

### 1.3 子表大小（BF16 2B） vs 鲲鹏 L1d=64K

子表改用 BF16（2B/entry）存储，大小减半，使 L1 驻留在更小的 G 值（更少 atomic 竞争）即可实现。

| G | step | 子表维度 | 子表大小 | L1 驻留 |
|---|------|---------|---------|---------|
| 1 | 256 | 256×256 | 128 KiB | ❌ L2 |
| **2** | **128** | **128×256** | **64 KiB** | **L1 边界** |
| **4** | **64** | **64×256** | **32 KiB** | **L1 ✅** |
| 8 | 32 | 32×256 | 16 KiB | L1 ✅ |
| 16 | 16 | 16×256 | 8 KiB | L1 ✅ |
| 32 | 8 | 8×256 | 4 KiB | L1 ✅ |
| 64 | 4 | 4×256 | 2 KiB | L1 ✅ |

> BF16 后 G=2 子表即达 L1 边界（64 KiB），G=4 起完全 L1 驻留。相比 float 版本（G=8 才进 L1），Atomic 竞争大幅降低。

---

## 2. 预处理

### 2.1 组划分

对每个 `A[i][k]`（uint8_t），确定所属组：

```
g = A[i][k] / (256 / G)
```

例如 G=4 时，step=64：
- idxA ∈ [0,63] → 组 0
- idxA ∈ [64,127] → 组 1
- idxA ∈ [128,191] → 组 2
- idxA ∈ [192,255] → 组 3

### 2.2 数据结构

```
                    按组重排后:
                    ┌──────────────────────────┐
A[i]:               │ 组0元素  │组1│组2│组3│  │
                    └──────────────────────────┘
                     ↑
              row_start[i*G+g] → 组 g 起始位置
              row_count[i*G+g] → 组 g 元素个数

A_grouped[total]      : 重排后的 A 值（组内连续）
k_indices[total]      : 每个元素对应的原始 k
```

### 2.3 存储开销

| 项目 | 大小 |
|------|------|
| `row_start` / `row_count` | N×G×4×2 = 128×G×8 B |
| `A_grouped` | N×L = 128×512 = 64 KiB |
| `k_indices` | N×L×4 = 256 KiB |
| **总计（G=64）** | **~384 KiB** |

`preprocess_groups()` 函数（第 636-665 行）一次预处理，供多次运行复用。

---

## 3. 核函数设计

### 3.1 两阶段 SVE 查表

鲲鹏 SVE 没有字节级 gather 指令（无法直接从 `B_T[j][k]` 加载单个字节到向量），所以采用**两阶段**设计：

```
阶段1（标量）        阶段2（SVE 向量化）
─────────────────────────────────────────────────
for t in 0..count:   svld1ub_u32 (A 值)
  b_local[t] =       svld1ub_u32 (B 值)
    B_T[j][k_ptr[t]]  → idx = (A-base)<<8 | B
                      → svld1_gather_u32index_f32
                        (子表, idx)  全部 L1 命中！
```

**为什么不在阶段 1 也用 SVE？**

不能将 `k_ptr[t]` 作为 SVE gather 的字节偏移来加载 `B_T[j]` 的单个字节。SVE 的 gather 指令最小粒度是 32-bit元素。所以阶段 1 用标量循环收集 B_T 值到局部数组 `b_local[512]`（栈上，L1），阶段 2 再将连续的 `b_local` 用 `svld1ub_u32` 加载。

### 3.2 源码：`lookup_sve_grouped_omp()`（第 666-723 行）

```cpp
#pragma omp parallel num_threads(G)
{
    int g = omp_get_thread_num();
    const float *sub = subtables[g].data();
    int base = g * step;

    for (int i = 0; i < N; ++i) {
        int start = data.row_start[i * G + g];
        int count = data.row_count[i * G + g];
        const uint8_t *a_ptr = data.A_grouped.data() + start;
        const int *k_ptr = data.k_indices.data() + start;

        for (int j = 0; j < S; ++j) {
            const uint8_t *b_row = B_T + j * L;

            // 阶段1: 收集 B_T 值（标量按 k 索引随机加载）
            alignas(16) uint8_t b_local[512];
            for (int t = 0; t < count; ++t)
                b_local[t] = b_row[k_ptr[t]];

            // 阶段2: SVE 向量化查子表（全部 L1 命中）
            svfloat32_t acc = svdup_n_f32(0.0f);
            int t = 0;
            svbool_t pg = svwhilelt_b32(t, count);
            while (svptest_any(svptrue_b32(), pg)) {
                svuint32_t a_vals = svld1ub_u32(pg, a_ptr + t);
                svuint32_t b_vals = svld1ub_u32(pg, b_local + t);
                svuint32_t idx = svorr_u32_z(pg,
                    svlsl_n_u32_z(pg, svsub_u32_z(pg, a_vals, base), 8),
                    b_vals);
                acc = svadd_f32_z(pg, acc,
                    svld1_gather_u32index_f32(pg, sub, idx));
                t += svcntw();
                pg = svwhilelt_b32(t, count);
            }
            #pragma omp atomic
            C[i * S + j] += svaddv_f32(svptrue_b32(), acc);
        }
    }
}
```

### 3.3 计时段

每个核独立计时，分离计算时间与同步时间：

```
t0         计算阶段（含 atomic）          t1           barrier     t2
├─────────────────────────────────────┤           ├────────────┤
│                                     │           │            │
│ for i,j:                            │           │  等待最慢核  │
│   阶段1 + 阶段2 + atomic             │           │            │
│                                     │           │            │
└─────────────────────────────────────┘           └────────────┘
   → core_times[g] = (t1 - t0) * 1e6 us            → sync_time
```

**输出解读：**

| 指标 | 含义 |
|------|------|
| `compute(min~max)` | G 个核各自的完成时间。范围越大负载越不均 |
| `sync` | barrier 等待时间（负载不均 + 系统调度） |
| `total = max(compute) + sync` | wall-clock 总耗时 |
| `GOP/s = N×S×L / total / 1e3` | 总吞吐 |

### 3.4 子表索引计算

```cpp
// subtable_g 是从完整 LUT 切出的连续区域，尺寸 step×256
// 索引公式: (a_val - base) * 256 + b_val
//   a_val - base  → 组内偏移 [0, step)
//   * 256         → 行内 b 维度
//   + b_val       → b 列的偏移

build_subtables(G, subtables, lut) {
    for g in 0..G:
        for a in 0..step:
            for b in 0..256:
                sub_g[a*256 + b] = lut[(g*step + a)*256 + b]
}
```

---

## 4. 实验设计

### 4.1 固定条件

| 参数 | 值 |
|------|-----|
| 矩阵 | N=128, S=328, L=512（与 x86 一致，可横向对比）|
| 线程 | 80（1 NUMA node, `close` 绑定）|
| LUT 格式 | BF16（2B，精度模拟）|
| 基线 | `lookup_sve_omp()` — SVE + 80 核行并行，无分组 |

### 4.2 测试变量

| G | 子表大小 | 使用核数 | 预期 Cache | 期望的收益 vs 基线 |
|---|---------|---------|-----------|------------------|
| 1 | 256K | 1 | L2 | 单核参考（无并行，验证正确性）|
| 2 | 128K | 2 | L2 | 弱并行，子表仍 > L1 |
| 4 | **64K** | 4 | **L1 边界** | 刚好满 L1，观察竞争影响 |
| 8 | 32K | 8 | **L1** | 预期拐点，L1 收益开始体现 |
| 16 | 16K | 16 | **L1** | 计算量减半，atomic 竞争加剧 |
| 32 | 8K | 32 | **L1** | 每核仅 ~16 个元素 |
| 64 | 4K | 64 | **L1** | 每核 ~8 元素，atomic 主导 |

### 4.3 实际结果

```
实验6: L1 分组建表查表试验 (SVE, 80 核)
矩阵: 128×512 * 328×512^T
L1d = 64 KiB, 子表格式: BF16 (2B)
Warmup=100, 采样=1000

方案    线程  子表        级别  计算(核min~max)同步      预处理     L1-miss%    L2-miss%    总耗时mean±sd 总耗时minGOP/s     正确
------------------------------------------------------------------------------------------------------------------------------------------------------
基线SVE 80      256 KiB       L2      —               —         —           —         —         538.9             —         39.89     OK
G=1       1       128 KiB       L2      27894.1~27894.1   0.9         427.6         8.95%       0.05%       28152.8±179.0    27894.6     0.77      OK
G=2       2       64 KiB        L1      12849.4~12882.4   24.7        462.0         1.75%       0.10%       13517.1±276.6    12915.6     1.66      OK
G=4       4       32 KiB        L1      6357.0~6598.0     124.1       354.6         0.12%       0.49%       7176.0±351.1     6601.2      3.26      OK
G=8       8       16 KiB        L1      3448.9~3616.2     53.3        386.1         0.12%       1.48%       3977.1±224.9     3621.6      5.94      OK
G=16      16      8 KiB         L1      2213.9~2592.9     716.1       358.2         0.13%       0.65%       3542.0±613.4     2595.4      8.28      OK
G=32      32      4 KiB         L1      1329.7~1919.7     549.7       363.4         1.42%       0.92%       2719.4±493.9     1939.8      11.08     OK
G=64      64      2 KiB         L1      1056.4~2430.4     371.2       365.5         6.02%       1.41%       2893.7±280.4     2431.1      8.84      OK
```

### 4.4 L1-miss% 验证

| G | 子表大小 | 理论 Cache | 实测 L1-miss% | 结论 |
|---|---------|-----------|--------------|------|
| 1 | 128 KiB | L2 | **8.95%** | L2，每次查表 L1 不命中 |
| 2 | 64 KiB | L1 边界 | **1.75%** | 大部分 L1 命中，边界效应 |
| 4 | 32 KiB | L1 | **0.12%** | 几乎全 L1 命中 |
| 8 | 16 KiB | L1 | **0.12%** | 全 L1 命中 |
| 16 | 8 KiB | L1 | **0.13%** | 全 L1 命中 |
| 32 | 4 KiB | L1 | **1.42%** | 多数 L1 命中，atomic 竞争引起少量额外 miss |
| 64 | 2 KiB | L1 | **6.02%** | 子表 L1 驻留，但 64 核缓存一致性流量导致 L1 污染 |

L1-miss% 数据证实了理论推测：G=1（128 KiB）超出 L1d 容量，8.95% 的 L1 miss；G=4 起子表完全驻留 L1，miss 率 < 0.2%。G=64 的 6.02% 异常升高值得关注——可能来自 64 核的 MESI 一致性流量导致 L1 行失效。

### 4.5 实际趋势

```
GOP/s
^
│                          基线 ─── 39.89
│
│                                          G=32 ← 最优
│                                        11.08
│                                      ↗
│                            G=16     ↗
│                           8.28     ↗
│                         ↗         ↘ G=64
│                G=8     ↗          8.84
│               5.94    ↗
│             ↗       ↗
│     G=4   ↗       ↗
│    3.26  ↗     ↗
│         ↗   ↗
│   G=2  ↗ ↗
│  1.66 ↗
│ G=1 ↗
│0.77
└───────────────────────────────────────→ G
   1   2   4   8   16   32   64
```

关键变化：
- **G=16 超过 G=8**（8.28 vs 5.94 GOP/s），虽然 atomic 竞争加剧但子表 L1 命中的收益更大
- **G=32 是最优值（11.08 GOP/s）**，sync 占比降至 549.7 μs（最佳平衡点）
- **G=64 反而下降（8.84 GOP/s）**，负载不均（1056~2430 μs，2.3×）和 L1 污染开始主导
- **基线 SVE OMP（39.89 GOP/s）仍远超所有 G 值**，atomic 方案无法与之竞争

---

## 5. 代码结构

```
matmul_fp8_server.cpp
├── 数据生成: gen_lut(), fill_random()
├── 计算核心（已有）
│   ├── lookup_scalar()        — 标量基线
│   ├── lookup_sve()           — SVE gather（串行）
│   ├── lookup_sve_omp()       — SVE + OpenMP
│   └── lookup_sve_tiled_omp() — SVE + 分块
├── 计算核心（新, 第636-723行）
│   ├── preprocess_groups()    — 按 idxA 分组预处理
│   ├── build_subtables_bf16() — 切分 BF16 子表
│   └── lookup_sve_grouped_omp() — 标量 BF16 查表 + SVE 累加
├── 实验函数（已有）
│   ├── exp1~5
└── 实验函数（新, 第725-835行）
    └── exp6_l1_grouped_lut()  — L1 分组建表扫描
```

### 5.1 编译

```bash
g++ -O3 -fopenmp -march=armv8.2-a+sve -o fp8_server \
    matmul_fp8_server.cpp -std=c++17 -lnuma
```

> 不需要额外的 fp16 编译选项。子表使用 float（4B），与已有 LUT 一致。

### 5.2 运行

```bash
numactl --cpunodebind=0 --membind=0 \
  OMP_NUM_THREADS=80 OMP_PLACES=cores OMP_PROC_BIND=close \
  ./fp8_server
```

---

## 6. 关键技术权衡

| 问题 | 影响 |
|------|------|
| **两阶段开销** | 阶段 1 的 B_T 标量收集增加一次循环（count 次），但 B_T 行 512B 在 L1，开销低 |
| **`#pragma omp atomic` 竞争** | G 个核同时写同一 C[i][j]，缓存行 ownership 频繁传递。G>8 时可能主导 |
| **负载不均** | 每组元素数由随机数据分布决定，各核 count 有差异（~±20%），最慢核限速 |
| **子表 = L1d 边界（G=4）** | 子表 64K 刚好等于 L1d，代码和数据可能挤占 L1I 或栈，实际效果打折 |
| **实验 6 与实验 2/3 对比** | 基线 SVE OMP（无分组）用 80 核 × 全表 L2 vs 分组用 G 核 × 子表 L1 + atomic |
