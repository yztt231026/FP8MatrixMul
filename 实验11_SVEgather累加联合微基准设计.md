# 实验11：SVE gather+累加联合微基准

## 1. 背景与目的

### 1.1 问题

实验10 将查表流程拆为三阶段（标量查表 → SVE 累加 → 归约），但 `lookup_sve` 的实际实现是 **SVE gather + 累加联合循环**（`svld1_gather_u32index_f32` 结果直接累加到 `acc`，无 `local_vals` 中间缓冲区）。

实验10 的三阶段拆分引入了额外的 store + load（向 `local_vals` 写再读），不能反映真实性能。

### 1.2 目标

匹配 `lookup_sve` 的真实实现模式：**SVE gather + 累放合并为一个阶段**，同时使用 `svadd_f32_z`（zeroing predicate，与 `lookup_sve` 一致）。

---

## 2. 与实验10的区别

| 维度 | 实验10 | 实验11 |
|------|--------|--------|
| 阶段数 | 3 阶段 | **2 阶段** |
| 数据流 | `gather → local_vals[] → svld1 → acc` | **gather 结果直接 → acc** |
| 查表方式 | 标量 `sub[indices[i]]` | **SVE `svld1_gather_u32index_f32`** |
| 累加谓词 | `svadd_f32_m`（merge） | **`svadd_f32_z`**（zeroing，匹配 lookup_sve） |
| 中间缓冲区 | `local_vals[]` 写 + 读 | **无** |

```
实验10:  sub[indices[i]]  →  local_vals[]  →  svld1 + svadd  →  svaddv
          标量查表（store）      SVE 累加（load）        归约

实验11:  svld1_gather + svadd_f32_z 联合循环  →  svaddv
          无中间缓冲区，gather 结果直接进 acc          归约
```

---

## 3. 测量方法

**计时方式**：`high_resolution_clock`（ns 精度），分 2 阶段计时。

| # | 阶段 | 伪代码 | 说明 |
|---|------|--------|------|
| 1 | SVE gather + 累加 | `svld1_gather + svadd_f32_z` 循环 | 匹配 lookup_sve |
| 2 | SVE 归约 | `svaddv_f32(acc)` | 水平归约 |

---

## 4. 实验设置

| 参数 | 值 |
|------|-----|
| 数据格式 | float（4B/entry），预计算索引 |
| 表维度扫描 | 2×256, 4×256, 8×256, 16×256, 32×256 |
| 查表次数 | `128 / (table_a / 2)` |
| 预热 | 2000 次 |
| 采样 | 20000 次 |
| 平台 | Kunpeng 920, 256-bit SVE |

---

## 5. 输出格式

```
表维度          表大小(KiB)    查表次数        平均耗时(μs)     总查表(ns)      每次查表(ns)    L1-miss%       gather+累加(μs)  归约(μs)         vs实验10
----------------------------------------------------------------------------------------------------------------------------------------------------
2×256           2              128             x.xxxx           xx.xx           xx.xx           x.xx%           x.xxxx            x.xxxx           —
4×256           4              64              x.xxxx           xx.xx           xx.xx           x.xx%           x.xxxx            x.xxxx           —
8×256           8              32              x.xxxx           xx.xx           xx.xx           x.xx%           x.xxxx            x.xxxx           —
16×256          16             16              x.xxxx           xx.xx           xx.xx           x.xx%           x.xxxx            x.xxxx           —
32×256          32             8               x.xxxx           xx.xx           xx.xx           x.xx%           x.xxxx            x.xxxx           —
```

**列说明**：

| 列 | 含义 |
|----|------|
| 表维度 | `table_a × 256` |
| 表大小(KiB) | entries × 4B / 1024 |
| 查表次数 | 每轮循环查表元素数 N |
| 平均耗时(μs) | 两阶段总耗时 / ITERS |
| gather+累加(μs) | `svld1_gather + svadd_f32_z` 联合循环耗时 |
| 归约(μs) | `svaddv_f32` 耗时 |
| vs实验10 | 同维度下与实验10 总耗时的对比 |

---

## 6. 预期分析

- **实验11 总耗时 ≤ 实验10 总耗时**，因为省去了 `local_vals` 的 store + load
- 但随着表变大，gather 的 L1-miss 增加，优势会被 cache miss 成本掩盖
- `svadd_f32_z` vs `svadd_f32_m`（实验10 使用 _m）：_z 不需要读目标寄存器旧值，理论上稍快，但差异很小

---

## 7. 代码索引

| 符号 | 说明 |
|------|------|
| `exp11_sve_gather_microbench()` | 主函数：SVE gather+累加联合循环 + 归约 |

### 编译运行

```bash
g++ -O3 -fopenmp -march=armv8.2-a+sve -o fp8_server matmul_fp8_server.cpp -std=c++17 -lnuma
./fp8_server 11
```
