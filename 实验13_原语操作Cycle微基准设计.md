# 实验13：原语操作 Cycle 微基准

## 1. 背景与目的

### 1.1 问题

实验 6~12 测量的是整体核函数或子流程的耗时（μs），但混淆了指令延迟、缓存命中、流水线并行等因素。当观察到一个加速比时，无法判断收益来自哪里：

- SVE gather 比标量查表到底快多少 cycle？
- `svaddv_f32` 归约本身消耗多少 cycle？
- 索引构建 `(a << 8) | b` 的成本是否可以忽略？
- 预处理重排的 scatter 写是否成为瓶颈？

### 1.2 目标

将 FP8 查表矩阵乘法的关键原语操作拆解出来，用 **CPU cycle 计数器** 独立测量每条指令/每个阶段的延迟：

- 隔离缓存层次的影响（L1 hit vs L2 hit）
- 隔离指令类型的影响（标量 load vs SVE gather vs SVE 顺序 load）
- 暴露预处理阶段的内存瓶颈

---

## 2. 实验设计

### 2.1 测量方法

**Cycle 计数器**：通过 Linux `perf_event_open` 系统调用打开 `PERF_COUNT_HW_CPU_CYCLES`，在测量循环开始前 enable、结束后 disable，累计读取总 cycles。

**测量模式**：每种原语操作在**单核**上循环执行 N 次（N=2000~50000），取 `总cycles / (N × 每批操作数)` 为平均每操作 cycles。

**"操作"定义**：
- **标量模式**：1 操作 = 1 次标量 load / 1 次索引计算（处理 1 个元素）
- **SVE 模式**：1 操作 = 1 条 SVE 向量指令（处理 `svcntw()=8` 个元素）
- Part 2 的归约和 Part 4 的预处理按实际指令计数

这样报告每操作 cycles 反映了**每条指令的实际延迟**，消除了向量宽度不同带来的摊销假象。

**空循环 overhead**：由循环体本身的指令（分支、`sink` 累加）贡献固定 overhead，各测量项的 overhead 近似相同，横向对比时可以抵消。

**Warmup**：正式测量前运行 100+ 次，确保缓存状态稳定、分支预测器 warmed up。

### 2.2 平台

| 参数 | 值 |
|------|-----|
| CPU | HiSilicon Kunpeng 920 |
| SVE | 256-bit (`svcntw()=8`) |
| L1d | 64 KiB |
| L2 | 1280 KiB |
| 频率 | ~2.6 GHz |
| 测量 | `perf_event_open` / `PERF_COUNT_HW_CPU_CYCLES` |

---

## 3. 四个测量部分

### 3.1 Part 1：标量 vs SVE load/store（L1/L2 查表）

测量内存加载指令本身的 cycle 成本，区分顺序访问和随机 gather。

#### 表参数

| 表 | 大小 | 层次 |
|----|------|------|
| L1 表 | 4 KiB (1024 floats) | 全部 L1 hit |
| L2 表 | 256 KiB (65536 floats) | L1 miss ~100%, L2 hit |

#### 访问模式（各 2 种查表次数：128、512）

| # | 模式 | 伪代码 | 说明 |
|---|------|--------|------|
| 1a | 标量顺序 | `for i: sum += tab[i]` | 连续地址，标量 load |
| 1b | SVE 顺序 | `svld1_f32 + svadd_f32_m` | 连续地址，SVE vector load |
| 1c | 标量 gather | `for i: sum += tab[rnd_idx[i]]` | 随机地址，标量 load |
| 1d | SVE gather | `svld1_gather + svadd_f32_m` | 随机地址，SVE gather 指令 |
| 1e | BF16 chunk (旧) | `u16→shift→store→svld1` | 仅在 L1 表测量，对照旧实现 |

#### 预期

- L1 标量顺序 ≈ 1~2 cyc/操作（L1 hit 顺序 prefetch）
- L1 SVE 顺序 ≈ 2~4 cyc/操作（`svld1_f32` 延迟，摊销前）
- L1 标量 gather ≈ 4~6 cyc/操作（L1 hit 但无 prefetch）
- L1 SVE gather ≈ 8~12 cyc/操作（`svld1_gather` 延迟）
- L2 标量 gather ≈ 10~15 cyc/操作（L2 延迟）
- BF16 chunk ≈ SVE gather + 额外 shift+store 开销

> 注意：SVE 1 次操作处理 8 个元素，所以 cyc/元素 = cyc/操作 ÷ 8。

---

### 3.2 Part 2：SVE sumup（归约）

测量向量归约指令的纯算术 cycle，无访存。

| # | 模式 | 向量数 | 说明 |
|---|------|--------|------|
| 2a | `svaddv_f32` | 1 | 单向量水平归约延迟 |
| 2b | `svadd_f32_m`×4 + 归约 | 4 | 匹配短累加链 |
| 2c | `svadd_f32_m`×16 + 归约 | 16 | 匹配中等累加链 |
| 2d | `svadd_f32_m`×64 + 归约 | 64 | 匹配长累加链（L=512） |

每操作 cycles = `总cycles / (重复次数 × (累加次数 + 1))`，1 次操作 = 1 条 `svadd_f32_m` 或 1 条 `svaddv_f32`。

#### 预期

- `svaddv_f32` ≈ 15~25 cyc/操作（SIMD 水平归约延迟较高）
- `svadd_f32_m` ≈ 2~4 cyc/操作（垂直加法可流水）

---

### 3.3 Part 3：索引构建（左移 + or）

这是 gather 前必须的标量/SVE 索引计算阶段。

| # | 模式 | 伪代码 | 说明 |
|---|------|--------|------|
| 3a | 标量 `(a<<8)\|b` | `idx = (a<<8)\|b` | 基础索引，一对 uint8 |
| 3b | 标量含偏移 | `idx = (a-base)*256+b` | 带减法偏移（分组子表） |
| 3c | SVE 向量化 | `svorr(svlsl(svld1ub,8), svld1ub)` | SVE 向量化构建，含 svst1 |

每批索引数：128、512。标量 1 操作 = 1 次索引计算，SVE 1 操作 = 1 次向量化构建（含 `svst1` 写回）。

#### 预期

- 标量 ≈ 1~2 cyc/操作
- SVE 向量化 ≈ 4~6 cyc/操作（`svld1ub`×2 + `svlsl` + `svorr` + `svst1`）

---

### 3.4 Part 4：预处理重排

测量 `preprocess_groups()` 中两个主要阶段（N=128, L=512）：

#### 4a：计数阶段

```cpp
row_count[i * G + A[i * L + k] / step]++;   // 直方图更新
```

每个元素引起一次 `row_count[]` 的**读-改-写**（L1 store）。对于随机 A 值，`g = idxA/step` 在组间近似均匀分布，`row_count` 分散在 N×G 个 counter 上。

#### 4b：填充阶段

```cpp
int g = A[k] / step;
int pos = row_start[i*G+g] + cursor[i*G+g];
A_grouped[pos] = A[k];
k_indices[pos] = k;
```

每个元素引起**两次写**（`A_grouped` + `k_indices`），写地址由 `cursor` 按组顺序推进，但不同的组间是**随机 scatter**（取决于 A 值）。

#### G 值扫描

| G | step | row_count / fill scatter 范围 | 预期 |
|---|------|------------------------------|------|
| 4 | 64 | N×G = 512 个 counter | 全部 L1 |
| 8 | 32 | 1024 个 counter | 全部 L1 |
| 16 | 16 | 2048 个 counter | 部分 L1 miss |
| 32 | 8 | 4096 个 counter | L1 可能溢出 |

---

## 4. 输出格式

```
══════════════════════════════════════════════════════════
实验13: 原语操作 cycle 微基准
平台: Kunpeng, svcntw()=8, L1d=64KiB, L2=1280KiB
══════════════════════════════════════════════════════════

— Part 1: 标量 vs SVE load/store (L1/L2 查表) —
  标量: 每操作 = 1 次标量 load | SVE: 每操作 = 1 次 SVE load/gather (8元素)
模式                      总cycles    每操作cycles
L1(4KiB) 标量顺序(128)    xxx         X.XX
L1(4KiB) SVE顺序(128)     xxx         X.XX
...

— Part 2: SVE sumup (归约) —
模式                      总cycles    每操作cycles
svaddv_f32 (1vec)         xxx         X.XX
...

— Part 3: 索引构建 (左移+or) —
  标量: 每操作 = 1 次索引计算 | SVE: 每操作 = 1 次向量化构建 (8索引)
模式                      总cycles    每操作cycles
...

— Part 4: 预处理重排 (G 值扫描，N=128, L=512) —
G        计数总cycles   填充总cycles   计数(cyc/op)   填充(cyc/op)
4         xxx           xxx            X.XX               X.XX
8         ...           ...            ...                ...
16        ...           ...            ...                ...
32        ...           ...            ...                ...
```

---

## 5. 代码索引

| 符号 | 说明 |
|------|------|
| `CycleCounter` | `perf_event_open` + `PERF_COUNT_HW_CPU_CYCLES` 封装 |
| `exp13_primitive_cycle_bench()` | 主函数：包含 Part 1~4 |

### 编译运行

```bash
g++ -O3 -fopenmp -march=armv8.2-a+sve -o fp8_server matmul_fp8_server.cpp -std=c++17 -lnuma
./fp8_server 13
```
