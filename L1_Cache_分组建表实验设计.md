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

### 1.3 子表大小（float 4B） vs 鲲鹏 L1d=64K

| G | idxA 范围 / 组 | 子表大小 | L1 驻留 |
|---|---------------|---------|---------|
| 1 | [0,255] | 256 KiB | ❌ L2 |
| 2 | [0,127] / [128,255] | 128 KiB | ❌ L2 |
| **4** | **[0,63] ... [192,255]** | **64 KiB** | **L1 边界** |
| 8 | [0,31] ... | **32 KiB** | **L1 ✅** |
| 16 | ... | 16 KiB | L1 ✅ |
| 32 | ... | 8 KiB | L1 ✅ |
| 64 | ... | 4 KiB | L1 ✅ |

> G=4 子表刚好等于 L1d 64K，可能有竞争。G=8 起子表足够小，L1 驻留确定。

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
| LUT 格式 | float（4B，复用已有 LUT）|
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

### 4.3 输出格式

```
实验6: L1 分组建表查表试验 (SVE, 80 核)
矩阵: 128×512 * 328×512^T
L1d = 64 KiB, 子表格式: float (4B)

方案    线程  子表        级别  计算(核min~max) 同步     总耗时(us) GOP/s     正确
---------------------------------------------------------------------------------------------------------------
基线SVE 80      256 KiB       L2      —               —         1070.0      20.11     OK
G=1       1       256 KiB       L2      53710.7~53710.7   2.5       53713.2      0.40      OK
G=2       2       128 KiB       L2      26860.1~26920.3   0.8       26921.1      0.80      OK
G=4       4       64 KiB        L1      13570.2~13810.5   1.2       13811.7      1.56      OK
G=8       8       32 KiB        L1       6810.4~7310.8    175.3      7486.1      2.87      OK
...
```

### 4.4 预期趋势

```
GOP/s
^
│                            G=4~8（L1 收益）
│                          ↗    ↘
│              G=2         ↗      ↘  G=16（atomic 竞争）
│            ↗           ↗         ↘
│  基线 ─●─●────●───●────────●───●───●──→
│       1   2   4   8   16   32   64   G
```

**关键假设：**

1. **G=4** 是 L1 驻留拐点（64K = L1d）。从 G=2→G=4 应有明显跳变（L2→L1 延迟减半）。
2. **G=8** 以上每核计算量减半但 atomic 竞争翻倍。在鲲鹏的缓存一致性协议下，atomic 开销可能低于 x86（MESI），G=8~16 仍是净收益区间。
3. **G=32~64** atomic 竞争可能抵消 L1 收益，出现拐点。

**与 x86 结果的预期差异：**

| | x86 i7-12700 (10核) | 鲲鹏 (80核) | 原因 |
|--|-------------------|------------|------|
| L1d | 48K → G=8 才进 L1 | 64K → **G=4 进 L1** | 子表更早 L1 驻留 |
| 核数 | 10（G 上限 8） | 80（G 上限 64） | 更大扫描范围 |
| atomic 竞争 | 严重（G=8 时 sync=361us） | 待测 | 不同缓存一致性协议 |
| 总核数 | 10 | 80 | 基线本身更快 |

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
│   ├── build_subtables()      — 切分子表
│   └── lookup_sve_grouped_omp() — SVE 分组查表（两阶段）
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
