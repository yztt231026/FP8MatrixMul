# 实验12：标量 vs SVE 查表分阶段对比

## 1. 背景与目的

### 1.1 问题

实验7/10/11 固定查表次数（最多至 128），无法观察**查表长度（K）变化**对各阶段耗时的影响。实际矩阵乘法的内循环长度 L=512，需要理解不同 K 下索引计算和查表各占多少比例。

### 1.2 目标

以 `lookup_scalar` / `lookup_sve` 两个函数为基准，将每个函数拆分为"索引计算"和"查表累加"两个阶段，K 从 128 到 65536 扫描，分析：

- 标量和 SVE 各自的瓶颈在哪一侧
- 加速比如何随 K 变化
- 索引计算在总耗时中的占比

---

## 2. 分解方案

### 标量分解

| # | 阶段 | 伪代码 | 说明 |
|---|------|--------|------|
| 1a | 完整流程 | `sum += table[(A[k]<<8) \| B[k]]` | 索引 + 查表 + 累加融合 |
| 1b | 索引计算 | `idx[k] = (A[k]<<8) \| B[k]` | 纯索引，写回内存 |
| 1c | 查表累加 | `sum += table[idx[k]]` | 用预计算索引，纯查表 |

### SVE 分解

| # | 阶段 | 伪代码 | 说明 |
|---|------|--------|------|
| 2a | 完整流程 | `svorr(svlsl(idxA,8), idxB) + gather + svadd + svaddv` | 匹配 lookup_sve |
| 2b | 索引计算 | `svorr(svlsl(idxA,8), idxB) + svst1` | 向量化索引，写回内存 |
| 2c | 查表累加 | `svld1_gather + svadd_f32_z` | 用预计算索引 |
| 2d | 归约 | `svaddv_f32` | 由 `total - idx - lu` 推算 |

**注意**：标量完整流程（1a）合并了索引和查表，索引结果在寄存器直接用作 table offset。拆分成 1b+1c 后中间结果必须落内存，**标量总可能小于标量索引+标量查表之和**——这是分阶段测量的 artifact。

---

## 3. 实验设置

| 参数 | 值 |
|------|-----|
| K 扫描 | 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536 |
| 查表 | 完整 LUT（256×256 float） |
| 预热 | 2000 次 |
| 采样 | 20000 次 |
| 平台 | Kunpeng 920, 256-bit SVE |

---

## 4. 输出格式

```
K            标量总(μs)     标量索引(μs)   标量查表(μs)    SVE总(μs)      SVE索引(μs)    SVE查表(μs)    SVE归约(μs)    总加速比
----------------------------------------------------------------------------------------------------------------------------------
128           x.xxx          x.xxx          x.xxx          x.xxx          x.xxx          x.xxx          x.xxx          xx.xx×
256           x.xxx          x.xxx          x.xxx          x.xxx          x.xxx          x.xxx          x.xxx          xx.xx×
512           x.xxx          x.xxx          x.xxx          x.xxx          x.xxx          x.xxx          x.xxx          xx.xx×
...
65536         x.xxx          x.xxx          x.xxx          x.xxx          x.xxx          x.xxx          x.xxx          xx.xx×
```

**列说明**：

| 列 | 含义 |
|----|------|
| K | 内循环长度（查表元素数） |
| 标量总(μs) | 完整标量流程：`sum += table[(A[k]<<8)\|B[k]]` |
| 标量索引(μs) | 纯索引计算：`idx[k] = (A[k]<<8)\|B[k]` |
| 标量查表(μs) | 预计算索引查表：`sum += table[idx[k]]` |
| SVE总(μs) | 完整 SVE 流程：`svlsl + svorr + svld1_gather + svadd + svaddv` |
| SVE索引(μs) | SVE 向量化索引计算 + svst1 写回 |
| SVE查表(μs) | 预计算索引，SVE gather + svadd_f32_z + svaddv |
| SVE归约(μs) | `= SVE总 - SVE索引 - SVE查表`，即 svaddv_f32 的贡献 |
| 总加速比 | 标量总 / SVE总 |

---

## 5. 预期分析

- **K 越大 → SVE 加速比越高**：向量化计算被摊销得越充分
- **标量总 < 标量索引 + 标量查表**：分阶段 artifact，因为拆开写回 idx_buf 增加了 store + 额外的 address dependency
- **SVE 索引占比极低**：`svlsl + svorr` 比标量移位的 `<<8 |` 快得多
- **SVE 查表为主瓶颈**：`svld1_gather` 的 L1-miss 随 K 变化不大（全 LUT 256 KiB，远大于 L1），但访问次数线性增加

---

## 6. 代码索引

| 符号 | 说明 |
|------|------|
| `exp12_lookup_phase_bench()` | 主函数：K 扫描 + 标量/SVE 分阶段对比 |

### 编译运行

```bash
g++ -O3 -fopenmp -march=armv8.2-a+sve -o fp8_server matmul_fp8_server.cpp -std=c++17 -lnuma
./fp8_server 12
```
