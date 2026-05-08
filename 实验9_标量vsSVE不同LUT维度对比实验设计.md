# 实验 9：标量 vs SVE gather — 不同 LUT 维度对比

## 1. 目的

对比两种查表矩阵乘法实现在不同 LUT 大小下的性能：
- **标量版**：顺序加载 A[i][k] 和 B_T[j][k]，`lut[idxA * dim + idxB]` 查表，标量累加
- **SVE gather 版**：SVE 字节加载 idxA/idxB，`svmul+svadd` 计算偏移，`svld1_gather_u32index_f32` gather 查表，SVE 向量化累加

观察 LUT 从 L2（256 KiB）缩小到 L1（1 KiB）时，SVE gather 相对标量的加速比变化。

## 2. 测试配置

| 参数 | 值 |
|------|-----|
| 矩阵 | N=128, S=328, L=512 |
| LUT 格式 | float（4B/entry） |
| LUT 维度 | 256×256, 128×128, 64×64, 32×32, 16×16 |
| LUT 大小 | 256 KiB → 1 KiB |
| 数据约束 | A/B_T 值域 [0, dim-1]，生成时已约束 |
| 索引计算 | `idxA * dim + idxB`（dim 为 LUT 边长） |
| 线程 | 1（单核） |
| 预热 | 20 次 |
| 采样 | 100 次取平均 |
| 计时 | `high_resolution_clock` 包裹完整矩阵乘 |

## 3. 预期

| LUT 维度 | LUT 大小 | Cache | 预期标量(μs) | 预期 SVE(μs) | 预期加速比 |
|---------|---------|-------|------------|------------|---------|
| 256×256 | 256 KiB | L2 | ~60000 | ~25000 | ~2.4× |
| 128×128 | 64 KiB | L1 边界 | ~25000 | ~15000 | ~1.7× |
| 64×64 | 16 KiB | L1 | ~18000 | ~13000 | ~1.4× |
| 32×32 | 4 KiB | L1 | ~16000 | ~12000 | ~1.3× |
| 16×16 | 1 KiB | L1 | ~15000 | ~11500 | ~1.3× |

SVE gather 的优势在 L2 时最大（gather 隐藏 L2 延迟），L1 全命中时优势缩小。

## 4. 核函数

### 标量版
```cpp
for (int k = 0; k < L; ++k)
    sum += lut[rowA[k] * dim + rowB[k]];
```

### SVE gather 版
```cpp
svuint32_t idxA = svld1ub_u32(pg, &rowA[k]);
svuint32_t idxB = svld1ub_u32(pg, &rowB[k]);
svuint32_t idx = svadd_u32_z(pg, svmul_n_u32_z(pg, idxA, dim), idxB);
acc = svadd_f32_z(pg, acc, svld1_gather_u32index_f32(pg, lut.data(), idx));
```

## 5. 输出格式

```
LUT 维度    LUT 大小      Cache 层级     标量(μs)       SVE gather(μs)   加速比
256×256     256 KiB       L2              xxx.x          xxx.x             x.xx×
128×128     64 KiB        L1              xxx.x          xxx.x             x.xx×
...
```

## 6. 代码索引

| 函数 | 说明 |
|------|------|
| `exp9_matrix_load_microbench()` | 主函数：LUT 维度扫描 + 标量/SVE 对比 |
| `bench_us` | lambda 计时器：warmup + 多次采样取平均 |
