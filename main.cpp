#include <iostream>
#include <vector>
#include <fstream>
#include <numeric>
// #include <arm_neon.h>
#include <arm_sve.h>
// 假设 g_S 为 256
const int g_N = 128;
const int g_L = 512;
const int g_S = 328;

// 加载二进制文件函数
template <typename T>
void load_bin(const std::string &path, T *data, size_t size)
{
    std::ifstream is(path, std::ios::binary);
    if (!is.is_open()) {
        std::cerr << "failed to open file:" << path << std::endl;
        return;
    }
    is.read(reinterpret_cast<char *>(data), size * sizeof(T));
}

#include <cstdint>
#include <cstdio>
#include <cstring>

// ─── 动态指令计数器（方案 C：代码插桩） ─────────────────────
struct KernelStats {
    const char *name;
    uint64_t loop_iters;       // 热循环迭代次数
    uint64_t elem_processed;   // 处理的元素总数（= loop_iters × 向量宽度）
    uint64_t reduce_count;     // 归约操作次数
    uint64_t gather_count;     // gather/查表指令数（FP8 专用）
    uint64_t compute_count;    // 核心计算指令数（fmla/sdot/smmmla）
};

#define PRINT_KERNEL_STATS(s) do { \
    printf("  %-40s  loop_iters=%10lu  elem=%10lu  reduce=%5lu  gather=%8lu  compute=%8lu\n", \
        (s).name, (s).loop_iters, (s).elem_processed, (s).reduce_count, (s).gather_count, (s).compute_count); \
} while(0)

#define RESET_STATS(s) do { \
    (s).loop_iters = 0; (s).elem_processed = 0; \
    (s).reduce_count = 0; (s).gather_count = 0; (s).compute_count = 0; \
} while(0)

static KernelStats g_stats_fp8_scalar  = {"matmul_fp8_lookup_scalar", 0,0,0,0,0};
static KernelStats g_stats_fp8_sve     = {"matmul_fp8_lookup_sve", 0,0,0,0,0};
static KernelStats g_stats_fp8_sve_opt = {"matmul_fp8_lookup_sve_optimized", 0,0,0,0,0};
static KernelStats g_stats_fp16        = {"matmul_sve_fp16", 0,0,0,0,0};
static KernelStats g_stats_i8_sve      = {"matmul_int8_sve", 0,0,0,0,0};
static KernelStats g_stats_i8mm        = {"matmul_int8_i8mm_complete", 0,0,0,0,0};

void print_all_stats() {
    const uint64_t total_ij = (uint64_t)g_N * g_S;  // 128 × 328 = 41,984
    const uint64_t total_calls = 10000;

    printf("\n" "══════════════════════════════════════════════════════════════════════════════════════\n");
    printf("动态指令计数报告（代码插桩）  N=%d S=%d L=%d  total_ij=%lu  loopCnt=%lu\n",
           g_N, g_S, g_L, total_ij, total_calls);
    printf("══════════════════════════════════════════════════════════════════════════════════════\n");
    printf("  %-40s %10s %10s %8s %8s %8s %8s\n",
           "Kernel", "loop_iters", "elem", "reduce", "gather", "compute", "iter/ij");
    printf("  ─────────────────────────────────────────────────────────────────────────────────────────\n");

    KernelStats *all[] = {&g_stats_fp8_scalar, &g_stats_fp8_sve, &g_stats_fp8_sve_opt,
                          &g_stats_fp16, &g_stats_i8_sve, &g_stats_i8mm};
    for (auto s : all) {
        uint64_t iters_per_ij = s->loop_iters / (total_calls * total_ij);
        printf("  %-40s %10lu %10lu %8lu %8lu %8lu %8lu\n",
               s->name, s->loop_iters, s->elem_processed,
               s->reduce_count, s->gather_count, s->compute_count, iters_per_ij);
    }
    printf("══════════════════════════════════════════════════════════════════════════════════════\n");

    printf("\n每call平均（÷%lu）：\n", total_calls);
    printf("  %-40s %10s %10s %8s %8s %8s\n", "Kernel", "loop_iters", "elem", "reduce", "gather", "compute");
    printf("  ───────────────────────────────────────────────────────────────────────\n");
    for (auto s : all) {
        printf("  %-40s %10lu %10lu %8lu %8lu %8lu\n",
               s->name,
               s->loop_iters / total_calls, s->elem_processed / total_calls,
               s->reduce_count / total_calls, s->gather_count / total_calls,
               s->compute_count / total_calls);
    }

    printf("\n每(ij)平均（÷%lu）：\n", total_calls * total_ij);
    printf("  %-40s %10s %10s %8s %8s %8s\n", "Kernel", "loop_iters", "elem", "reduce", "gather", "compute");
    printf("  ───────────────────────────────────────────────────────────────────────\n");
    uint64_t per_ij = total_calls * total_ij;
    for (auto s : all) {
        printf("  %-40s %10lu %10lu %8lu %8lu %8lu\n",
               s->name,
               s->loop_iters / per_ij, s->elem_processed / per_ij,
               s->reduce_count / per_ij, s->gather_count / per_ij,
               s->compute_count / per_ij);
    }
}

/**
 * 标量版 FP8 查表模拟矩阵乘法
 * @param table  256x256 的 FP16 查找表 (线性存储)
 * @param A      N*L 的 FP8 (uint8) 矩阵
 * @param B_T    S*L 的 FP8 (uint8) 矩阵 (已转置)
 * @param C      N*S 的 FP32 结果矩阵
 */
void matmul_fp8_lookup_scalar(const float *table, const uint8_t *A, const uint8_t *B_T, float *C, int N, int S, int L)
{
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < S; ++j) {
            float sum = 0.0f;
            const uint8_t *rowA = &A[i * L];
            const uint8_t *rowB = &B_T[j * L];
            for (int k = 0; k < L; ++k) {
                g_stats_fp8_scalar.loop_iters++;
                // 1. 获取两个矩阵中的 FP8 索引值
                uint8_t idxA = rowA[k];
                uint8_t idxB = rowB[k];

                // 2. 计算在查找表中的线性偏移地址 (row * 256 + col)
                // 使用位移运算 idxA << 8 替代 idxA * 256
                int offset = (static_cast<int>(idxA) << 8) | idxB;
                //  3. 查表并累加
                sum += static_cast<float>(table[offset]);
            }
            g_stats_fp8_scalar.elem_processed += L;
            g_stats_fp8_scalar.reduce_count++;

            // 4. 将结果存入 C 矩阵
            C[i * S + j] = sum;
        }
    }
}

/**
 * SVE 查表法模拟矩阵乘法 (Table 为 float)
 * @param table  256x256 的 float 查找表 (线性存储)
 * @param A      N*L 的 FP8 (uint8) 矩阵
 * @param B_T    S*L 的 FP8 (uint8) 矩阵 (已转置)
 * @param C      N*S 的 float 结果矩阵
 */
void matmul_fp8_lookup_sve(const float *table, const uint8_t *A, const uint8_t *B_T, float *C, int N, int S, int L)
{
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < S; ++j) {
            const uint8_t *rowA = A + i * L;
            const uint8_t *rowB = B_T + j * L;

            // 初始化 float 向量累加器
            svfloat32_t acc_v = svdup_n_f32(0.0f);

            int k = 0;
            // 步进使用 svcntw()，因为我们处理的是 32 位的 float 和 32 位的索引
            svbool_t pg = svwhilelt_b32(k, L);

            while (svptest_any(svptrue_b32(), pg)) {
                g_stats_fp8_sve.loop_iters++;
                // 1. 加载 8 位索引并扩展到 32 位 (u8 -> u32)
                svuint32_t idxA = svld1ub_u32(pg, &rowA[k]);
                svuint32_t idxB = svld1ub_u32(pg, &rowB[k]);

                // 2. 拼接成查表索引: (idxA << 8) | idxB
                svuint32_t indices = svorr_u32_z(pg, svlsl_n_u32_z(pg, idxA, 8), idxB);

                // 3. Gather 加载 float 数据
                svfloat32_t vals = svld1_gather_u32index_f32(pg, table, indices);
                g_stats_fp8_sve.gather_count++;

                // 4. 向量累加
                acc_v = svadd_f32_z(pg, acc_v, vals);

                k += svcntw();
                pg = svwhilelt_b32(k, L);
            }
            g_stats_fp8_sve.elem_processed += L;
            g_stats_fp8_sve.reduce_count++;

            // 5. 归约求和并存入结果矩阵
            C[i * S + j] = svaddv_f32(svptrue_b32(), acc_v);
        }
    }
}

void matmul_fp8_lookup_sve_optimized(
    const float *table, const uint8_t *A, const uint8_t *B_T, float *C, int N, int S, int L)
{
    // --- 1. 预处理 A 矩阵 ---
    // 申请内存将 A(u8) 转存为 A_shifted(u32), 每个元素左移 8 位
    // 建议使用对齐申请，有利于 SVE 加载性能
    uint32_t *A_shifted = (uint32_t *)aligned_alloc(64, N * L * sizeof(uint32_t));

    // 这里可以用简单循环，编译器通常会对此类简单移位进行自动向量化
    for (int i = 0; i < N * L; ++i) {
        A_shifted[i] = (uint32_t)A[i] << 8;
    }

    // --- 2. 矩阵乘法计算 ---
    for (int i = 0; i < N; ++i) {
        // 指向预处理后的 A 矩阵行
        const uint32_t *rowA_sh = A_shifted + i * L;

        for (int j = 0; j < S; ++j) {
            const uint8_t *rowB = B_T + j * L;
            svfloat32_t acc_v = svdup_n_f32(0.0f);

            int k = 0;
            svbool_t pg = svwhilelt_b32(k, L);

            while (svptest_any(svptrue_b32(), pg)) {
                g_stats_fp8_sve_opt.loop_iters++;
                // 直接加载已经左移好的 A (u32)
                svuint32_t va_sh = svld1_u32(pg, &rowA_sh[k]);

                // 加载 B 并拓宽 (u8 -> u32)
                svuint32_t vb = svld1ub_u32(pg, &rowB[k]);

                // 拼接索引：直接 OR 操作，省去了循环内的移位指令
                svuint32_t indices = svorr_u32_z(pg, va_sh, vb);

                // Gather 加载并累加
                svfloat32_t vals = svld1_gather_u32index_f32(pg, table, indices);
                g_stats_fp8_sve_opt.gather_count++;
                acc_v = svadd_f32_z(pg, acc_v, vals);

                k += svcntw();
                pg = svwhilelt_b32(k, L);
            }
            g_stats_fp8_sve_opt.elem_processed += L;
            g_stats_fp8_sve_opt.reduce_count++;
        }
    }

    // --- 3. 释放临时内存 ---
    free(A_shifted);
}

/**
 * 使用 SVE 计算两个 FP16 向量的点积并归约求和
 * @param a 向量 A
 * @param b 向量 B
 * @param n 元素个数
 * 编译参数: -march=armv8.2-a+sve+fp16
 */
float dotProductSVE(const __fp16 *a, const __fp16 *b, size_t n)
{
    // 累加器初始化为 0 (向量形式)
    svfloat16_t acc_v = svdup_n_f16(0.0f);

    // 生成控制循环的谓词，自动处理长度为 n 的边界
    size_t i = 0;
    svbool_t pg = svwhilelt_b16(i, n);

    // 当谓词 pg 中仍有 True 时继续循环
    while (svptest_any(svptrue_b16(), pg)) {
        // 1. 带谓词加载数据（只加载有效索引范围内的元素）
        svfloat16_t va = svld1_f16(pg, &a[i]);
        svfloat16_t vb = svld1_f16(pg, &b[i]);

        // 2. 乘加操作：acc_v = acc_v + (va * vb)
        // 使用 FMLA (Floating-point Multiply-Add) 指令
        acc_v = svmla_f16_z(pg, acc_v, va, vb);

        // 3. 更新索引并生成新的谓词
        i += svcnth();
        pg = svwhilelt_b16(i, n);
    }

    // 4. 向量归约求和：将向量内所有 FP16 通道的值相加
    // 返回值转为标量 float 以保证求和精度并方便后续使用
    return (float)svaddv_f16(svptrue_b16(), acc_v);
}

#include <arm_sve.h>
#include <vector>

/**
 * FP16 矩阵乘法（B 已预转置）
 * 结果矩阵 C = A * B^T, 维度为 N * S
 * @param A 矩阵 A (N * L)
 * @param B_T 矩阵 B 的转置 (S * L)
 * @param C 结果矩阵 C (N * S)
 */
void matmul_sve_fp16(const __fp16 *A, const __fp16 *B_T, float *C, int N, int S, int L)
{
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < S; ++j) {
            // 获取当前行向量的起始指针
            const __fp16 *rowA = A + i * L;
            const __fp16 *rowB = B_T + j * L;

            svfloat16_t acc_v = svdup_n_f16(0.0f);
            int k = 0;
            // 自动处理 L 不对齐的情况
            svbool_t pg = svwhilelt_b16(k, L);

            while (svptest_any(svptrue_b16(), pg)) {
                g_stats_fp16.loop_iters++;
                svfloat16_t va = svld1_f16(pg, &rowA[k]);
                svfloat16_t vb = svld1_f16(pg, &rowB[k]);

                // 乘加运算
                acc_v = svmla_f16_z(pg, acc_v, va, vb);
                g_stats_fp16.compute_count++;

                k += svcnth();
                pg = svwhilelt_b16(k, L);
            }
            g_stats_fp16.elem_processed += L;
            g_stats_fp16.reduce_count++;
        }
    }
}

template <class T, class U>
void matmul_scalar(const T *A, const T *B_T, U *C, int N, int S, int L)
{
    const T *rowA = A;
    U *res = C;
    for (int i = 0; i < N; ++i) {
        const T *rowB = B_T;
        for (int j = 0; j < S; ++j) {
            int32_t sum = 0;
            // const T *rowA = A + i * L;
            // const T *rowB = B_T + j * L;

            for (int k = 0; k < L; ++k) {
                // 将 int8 强转为 int32 再相乘累加
                sum += static_cast<U>(static_cast<U>(rowA[k]) * static_cast<U>(rowB[k]));
            }
            // C[i * S + j] = sum;
            *res = sum;
            res++;
            rowB += L;
        }
        rowA += L;
    }
}

/**
 * 标量版 INT8 矩阵乘法 (B 已转置)
 * C = A * B_T
 */
// void matmul_int8_scalar(const int8_t *A, const int8_t *B_T, int32_t *C, int N, int S, int L)
// {
//     for (int i = 0; i < N; ++i) {
//         for (int j = 0; j < S; ++j) {
//             int32_t sum = 0;
//             const int8_t *rowA = A + i * L;
//             const int8_t *rowB = B_T + j * L;

//             for (int k = 0; k < L; ++k) {
//                 // 将 int8 强转为 int32 再相乘累加
//                 sum += static_cast<int32_t>(static_cast<int32_t>(rowA[k]) * static_cast<int32_t>(rowB[k]));
//             }
//             C[i * S + j] = sum;
//         }
//     }
// }

/**
 * SVE 加速版 INT8 矩阵乘法 (B 已转置)
 * 编译参数: -march=armv8.2-a+sve+dotprod
 */
void matmul_int8_sve(const int8_t *A, const int8_t *B_T, int32_t *C, int N, int S, int L)
{
    // std::cout << "svcnth(); = " << svcnth() << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < S; ++j) {
            const int8_t *rowA = A + i * L;
            const int8_t *rowB = B_T + j * L;

            // 初始化 32 位有符号累加器向量为 0
            svint32_t acc_v = svdup_n_s32(0);

            int k = 0;
            // 步进处理 int8 数据
            svbool_t pg = svwhilelt_b8(k, L);

            while (svptest_any(svptrue_b8(), pg)) {
                g_stats_i8_sve.loop_iters++;
                // 1. 加载 int8 向量
                svint8_t va = svld1_s8(pg, &rowA[k]);
                svint8_t vb = svld1_s8(pg, &rowB[k]);

                // 2. 核心：点积指令
                acc_v = svdot_s32(acc_v, va, vb);
                g_stats_i8_sve.compute_count++;

                // 3. 步进：svcntb() 返回当前字节通道数
                k += svcntb();
                pg = svwhilelt_b8(k, L);
            }
            g_stats_i8_sve.elem_processed += L;
            g_stats_i8_sve.reduce_count++;
        }
    }
}

/**
 * 高性能 I8MM 矩阵乘法（带边界处理）
 * 编译：g++ -O3 -march=armv8.2-a+sve+i8mm
 */
void matmul_int8_i8mm_complete(const int8_t *A, const int8_t *B_T, int32_t *C, int N, int S, int L)
{
    int i = 0;
    // std::cout << "svcntb() = " << svcntb() << std::endl;
    //  1. 主循环：处理 2x2 的分块
    for (; i <= N - 2; i += 2) {
        int j = 0;
        for (; j <= S - 2; j += 2) {
            const int8_t *rA0 = A + (i + 0) * L, *rA1 = A + (i + 1) * L;
            const int8_t *rB0 = B_T + (j + 0) * L, *rB1 = B_T + (j + 1) * L;

            svint32_t acc00 = svdup_n_s32(0), acc01 = svdup_n_s32(0);
            svint32_t acc10 = svdup_n_s32(0), acc11 = svdup_n_s32(0);

            for (int k = 0; k < L; k += svcntb()) {
                g_stats_i8mm.loop_iters++;
                svbool_t pg = svwhilelt_b8(k, L);
                svint8_t va0 = svld1_s8(pg, &rA0[k]), va1 = svld1_s8(pg, &rA1[k]);
                svint8_t vb0 = svld1_s8(pg, &rB0[k]), vb1 = svld1_s8(pg, &rB1[k]);

                acc00 = svmmla_s32(acc00, va0, vb0);
                acc01 = svmmla_s32(acc01, va0, vb1);
                acc10 = svmmla_s32(acc10, va1, vb0);
                acc11 = svmmla_s32(acc11, va1, vb1);
                g_stats_i8mm.compute_count += 4;
            }
            g_stats_i8mm.elem_processed += L * 4;  // 2x2 output
            g_stats_i8mm.reduce_count += 4;
            C[(i + 0) * S + (j + 0)] = svaddv_s32(svptrue_b32(), acc00);
            C[(i + 0) * S + (j + 1)] = svaddv_s32(svptrue_b32(), acc01);
            C[(i + 1) * S + (j + 0)] = svaddv_s32(svptrue_b32(), acc10);
            C[(i + 1) * S + (j + 1)] = svaddv_s32(svptrue_b32(), acc11);
        }
        // 处理 S 为奇数时剩下的最后一列
        if (j < S) {
            const int8_t *rA0 = A + (i + 0) * L, *rA1 = A + (i + 1) * L, *rB = B_T + j * L;
            svint32_t acc0 = svdup_n_s32(0), acc1 = svdup_n_s32(0);
            for (int k = 0; k < L; k += svcntb()) {
                svbool_t pg = svwhilelt_b8(k, L);
                svint8_t vb = svld1_s8(pg, &rB[k]);
                acc0 = svmmla_s32(acc0, svld1_s8(pg, &rA0[k]), vb);
                acc1 = svmmla_s32(acc1, svld1_s8(pg, &rA1[k]), vb);
                g_stats_i8mm.compute_count += 2;
                g_stats_i8mm.loop_iters++;
            }
            g_stats_i8mm.elem_processed += L * 2;
            g_stats_i8mm.reduce_count += 2;
            C[(i + 0) * S + j] = svaddv_s32(svptrue_b32(), acc0);
            C[(i + 1) * S + j] = svaddv_s32(svptrue_b32(), acc1);
        }
    }

    // 2. 处理 N 为奇数时剩下的最后一行
    if (i < N) {
        for (int j = 0; j < S; ++j) {
            const int8_t *rA = A + i * L, *rB = B_T + j * L;
            svint32_t acc = svdup_n_s32(0);
            for (int k = 0; k < L; k += svcntb()) {
                svbool_t pg = svwhilelt_b8(k, L);
                acc = svmmla_s32(acc, svld1_s8(pg, &rA[k]), svld1_s8(pg, &rB[k]));
                g_stats_i8mm.compute_count++;
                g_stats_i8mm.loop_iters++;
            }
            g_stats_i8mm.elem_processed += L;
            g_stats_i8mm.reduce_count++;
            C[i * S + j] = svaddv_s32(svptrue_b32(), acc);
        }
    }
}

#include <chrono>
using TimoPoint = std::chrono::high_resolution_clock::time_point;

template <class T>
void PrintfVector(std::vector<T> &datas)
{
    for (int i = 0; i < datas.size(); ++i) {
        std::cout << datas[i] << ' ';
        if ((i + 1) % 16 == 0 || i == datas.size() - 1) {
            std::cout << std::endl;
        }
    }
}

template <class T>
T Average(std::vector<T> &datas)
{
    T sum = std::accumulate(datas.begin(), datas.end(), 0);
    return sum / datas.size();
}

int main()
{
    // 分配内存
    std::vector<uint8_t> a_fp8(g_N * g_L), b_fp8(g_S * g_L);
    std::vector<float> lut(256 * 256);
    std::vector<__fp16> a_fp16(g_N * g_L), b_fp16(g_S * g_L);
    std::vector<float> res(g_N * g_S);

    std::vector<int8_t> a_i8(g_N * g_L), b_i8(g_S * g_L);
    std::vector<int32_t> res_i32(g_N * g_S);

    // 读取 5 个文件
    load_bin("./input/matrix_a_8.bin", a_fp8.data(), g_N * g_L);
    load_bin("./input/matrix_b_8.bin", b_fp8.data(), g_S * g_L);
    load_bin("./input/look_up_table_fp32.bin", lut.data(), 256 * 256);
    load_bin("./input/matrix_a_16.bin", a_fp16.data(), g_N * g_L);
    load_bin("./input/matrix_b_16.bin", b_fp16.data(), g_S * g_L);
    load_bin("./input/matrix_a_i8.bin", a_i8.data(), g_N * g_L);
    load_bin("./input/matrix_b_i8.bin", b_i8.data(), g_S * g_L);

    int loopCnt = 10000;
    int warmup = 100;
    // 预热（所有kernel都跑一遍warmup次）
    std::cout << "Warming up (" << warmup << " iterations per kernel)..." << std::endl;
    for (int i = 0; i < warmup; ++i) {
        matmul_fp8_lookup_scalar(lut.data(), a_fp8.data(), b_fp8.data(), res.data(), g_N, g_S, g_L);
        matmul_fp8_lookup_sve(lut.data(), a_fp8.data(), b_fp8.data(), res.data(), g_N, g_S, g_L);
        matmul_fp8_lookup_sve_optimized(lut.data(), a_fp8.data(), b_fp8.data(), res.data(), g_N, g_S, g_L);
        matmul_sve_fp16(a_fp16.data(), b_fp16.data(), res.data(), g_N, g_S, g_L);
        matmul_scalar(a_i8.data(), b_i8.data(), res_i32.data(), g_N, g_S, g_L);
        matmul_int8_sve(a_i8.data(), b_i8.data(), res_i32.data(), g_N, g_S, g_L);
        matmul_int8_i8mm_complete(a_i8.data(), b_i8.data(), res_i32.data(), g_N, g_S, g_L);
    }
    // 复位计数器（预热不计入统计）
    RESET_STATS(g_stats_fp8_scalar);
    RESET_STATS(g_stats_fp8_sve);
    RESET_STATS(g_stats_fp8_sve_opt);
    RESET_STATS(g_stats_fp16);
    RESET_STATS(g_stats_i8_sve);
    RESET_STATS(g_stats_i8mm);

    {  // 查表计算
        std::cout << "\nComputing FP8 LUT MatMul..." << std::endl;
        std::vector<float> times(loopCnt);
        for (int i = 0; i < loopCnt; ++i) {
            TimoPoint tpBegin = std::chrono::high_resolution_clock::now();
            matmul_fp8_lookup_scalar(lut.data(), a_fp8.data(), b_fp8.data(), res.data(), g_N, g_S, g_L);
            TimoPoint tpEnd = std::chrono::high_resolution_clock::now();
            times[i] = (tpEnd - tpBegin).count() / 1000;
        }
        std::cout << "FP8 lookup scalar dura = " << Average(times) << " us" << std::endl;
        for (int i = 0; i < loopCnt; ++i) {
            TimoPoint tpBegin = std::chrono::high_resolution_clock::now();
            matmul_fp8_lookup_sve(lut.data(), a_fp8.data(), b_fp8.data(), res.data(), g_N, g_S, g_L);
            TimoPoint tpEnd = std::chrono::high_resolution_clock::now();
            times[i] = (tpEnd - tpBegin).count() / 1000;
        }
        std::cout << "FP8 lookup sve dura = " << Average(times) << " us" << std::endl;
        for (int i = 0; i < loopCnt; ++i) {
            TimoPoint tpBegin = std::chrono::high_resolution_clock::now();
            matmul_fp8_lookup_sve_optimized(lut.data(), a_fp8.data(), b_fp8.data(), res.data(), g_N, g_S, g_L);
            TimoPoint tpEnd = std::chrono::high_resolution_clock::now();
            times[i] = (tpEnd - tpBegin).count() / 1000;
        }
        std::cout << "FP8 lookup sve opt dura = " << Average(times) << " us" << std::endl;
        // std::chrono::nanoseconds dura1 = tpAfterFP8LookupScalar - tpBegin;
        // std::chrono::nanoseconds dura2 = tpAfterFP8LookupSve - tpAfterFP8LookupScalar;
        // std::chrono::nanoseconds dura3 = tpAfterFP8LookupSveOpt - tpAfterFP8LookupSve;
        // std::cout << "FP8 lookup scalar dura = " << dura1.count() / loopCnt / 1000.0 << " us" << std::endl;
        // std::cout << "FP8 lookup sve dura = " << dura2.count() / loopCnt / 1000.0 << " us" << std::endl;
        // std::cout << "FP8 lookup sve opt dura = " << dura3.count() / loopCnt / 1000.0 << " us" << std::endl;
    }
    {  // FP16 指令加速计算
        std::cout << "Computing FP16 sve MatMul..." << std::endl;
        TimoPoint tpBegin = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < loopCnt; ++i) {
            matmul_sve_fp16(a_fp16.data(), b_fp16.data(), res.data(), g_N, g_S, g_L);
        }
        TimoPoint tpAfterFP16SveMatMul = std::chrono::high_resolution_clock::now();
        std::chrono::nanoseconds dura1 = tpAfterFP16SveMatMul - tpBegin;
        std::cout << "FP16 sve mat mul dura = " << dura1.count() / loopCnt / 1000.0 << " us" << std::endl;
    }
    {  // i8矩阵乘法运算
        std::cout << "Computing i8 MatMul..." << std::endl;
        TimoPoint tpBegin = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < loopCnt; ++i) {
            matmul_scalar(a_i8.data(), b_i8.data(), res_i32.data(), g_N, g_S, g_L);
        }
        TimoPoint tpAfterI8ScalarMatMul = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < loopCnt; ++i) {
            matmul_int8_sve(a_i8.data(), b_i8.data(), res_i32.data(), g_N, g_S, g_L);
        }
        TimoPoint tpAfterI8SveMatMul = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < loopCnt; ++i) {
            matmul_int8_i8mm_complete(a_i8.data(), b_i8.data(), res_i32.data(), g_N, g_S, g_L);
        }
        TimoPoint tpAfterI8mmMatMul = std::chrono::high_resolution_clock::now();
        std::chrono::nanoseconds dura1 = tpAfterI8ScalarMatMul - tpBegin;
        std::chrono::nanoseconds dura2 = tpAfterI8SveMatMul - tpAfterI8ScalarMatMul;
        std::chrono::nanoseconds dura3 = tpAfterI8mmMatMul - tpAfterI8SveMatMul;

        std::cout << "i8 scalar mat mul dura = " << dura1.count() / loopCnt / 1000.0 << " us" << std::endl;
        std::cout << "i8 sve mat mul dura = " << dura2.count() / loopCnt / 1000.0 << " us" << std::endl;
        std::cout << "i8 mm mat mul dura = " << dura3.count() / loopCnt / 1000.0 << " us" << std::endl;
    }
    print_all_stats();
    return 0;
}