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
#include <string>
#include <functional>
#include <iomanip>
#include <omp.h>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>

// ─── PMU 性能计数器（perf_event_open） ─────────────────────
class PerfCounter {
    int fd_ = -1;
    static long sys_open(struct perf_event_attr *pea, pid_t pid, int cpu,
                         int group_fd, unsigned long flags) {
        return syscall(__NR_perf_event_open, pea, pid, cpu, group_fd, flags);
    }
public:
    PerfCounter(uint64_t config, bool exclude_kernel = true) {
        struct perf_event_attr pea{};
        pea.type = PERF_TYPE_RAW;
        pea.size = sizeof(pea);
        pea.config = config;
        pea.disabled = 1;
        pea.pinned = 1;
        pea.exclude_kernel = exclude_kernel ? 1 : 0;
        pea.exclude_hv = 1;
        fd_ = sys_open(&pea, 0, -1, -1, 0);
    }
    ~PerfCounter() { if (fd_ >= 0) close(fd_); }
    bool ok() const { return fd_ >= 0; }
    void enable()  { if (fd_ >= 0) ioctl(fd_, PERF_EVENT_IOC_ENABLE, 0); }
    void disable() { if (fd_ >= 0) ioctl(fd_, PERF_EVENT_IOC_DISABLE, 0); }
    void reset()   { if (fd_ >= 0) ioctl(fd_, PERF_EVENT_IOC_RESET, 0); }
    uint64_t read() {
        uint64_t val = 0;
        if (fd_ >= 0 && ::read(fd_, &val, sizeof(val)) == sizeof(val)) return val;
        return 0;
    }
};

/** ARMv8 PMU 事件编码（鲲鹏 920 兼容） */
namespace ArmPmu {
    constexpr uint64_t L1D_CACHE        = 0x04;
    constexpr uint64_t L1D_CACHE_REFILL = 0x03;
    constexpr uint64_t L2D_CACHE        = 0x16;
    constexpr uint64_t L2D_CACHE_REFILL = 0x17;
}

struct CacheCounters {
    uint64_t l1_access = 0, l1_refill = 0, l2_access = 0, l2_refill = 0;
    bool ok = false;
};

/** 运行 fn() repeat 次，累计 L1/L2 miss 计数并计算 miss 率 */
CacheCounters measure_cache(const std::function<void()> &fn, int warmup = 10, int repeat = 100) {
    CacheCounters cc;
    for (int i = 0; i < warmup; ++i) fn();

    PerfCounter c_l1a(ArmPmu::L1D_CACHE);
    PerfCounter c_l1r(ArmPmu::L1D_CACHE_REFILL);
    PerfCounter c_l2a(ArmPmu::L2D_CACHE);
    PerfCounter c_l2r(ArmPmu::L2D_CACHE_REFILL);
    if (!c_l1a.ok() || !c_l1r.ok() || !c_l2a.ok() || !c_l2r.ok())
        return cc;

    c_l1a.reset(); c_l1r.reset(); c_l2a.reset(); c_l2r.reset();
    c_l1a.enable(); c_l1r.enable(); c_l2a.enable(); c_l2r.enable();
    for (int i = 0; i < repeat; ++i) fn();
    c_l1a.disable(); c_l1r.disable(); c_l2a.disable(); c_l2r.disable();

    cc.l1_access = c_l1a.read();
    cc.l1_refill = c_l1r.read();
    cc.l2_access = c_l2a.read();
    cc.l2_refill = c_l2r.read();
    cc.ok = true;
    return cc;
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
                // 1. 获取两个矩阵中的 FP8 索引值
                uint8_t idxA = rowA[k];
                uint8_t idxB = rowB[k];

                // 2. 计算在查找表中的线性偏移地址 (row * 256 + col)
                // 使用位移运算 idxA << 8 替代 idxA * 256
                int offset = (static_cast<int>(idxA) << 8) | idxB;
                //  3. 查表并累加
                sum += static_cast<float>(table[offset]);
            }

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
                // 1. 加载 8 位索引并扩展到 32 位 (u8 -> u32)
                svuint32_t idxA = svld1ub_u32(pg, &rowA[k]);
                svuint32_t idxB = svld1ub_u32(pg, &rowB[k]);

                // 2. 拼接成查表索引: (idxA << 8) | idxB
                svuint32_t indices = svorr_u32_z(pg, svlsl_n_u32_z(pg, idxA, 8), idxB);

                // 3. Gather 加载 float 数据
                svfloat32_t vals = svld1_gather_u32index_f32(pg, table, indices);

                // 4. 向量累加
                acc_v = svadd_f32_z(pg, acc_v, vals);

                k += svcntw();
                pg = svwhilelt_b32(k, L);
            }

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
                // 直接加载已经左移好的 A (u32)
                svuint32_t va_sh = svld1_u32(pg, &rowA_sh[k]);

                // 加载 B 并拓宽 (u8 -> u32)
                svuint32_t vb = svld1ub_u32(pg, &rowB[k]);

                // 拼接索引：直接 OR 操作，省去了循环内的移位指令
                svuint32_t indices = svorr_u32_z(pg, va_sh, vb);

                // Gather 加载并累加
                svfloat32_t vals = svld1_gather_u32index_f32(pg, table, indices);
                acc_v = svadd_f32_z(pg, acc_v, vals);

                k += svcntw();
                pg = svwhilelt_b32(k, L);
            }
            C[i * S + j] = svaddv_f32(svptrue_b32(), acc_v);
        }
    }

    // --- 3. 释放临时内存 ---
    free(A_shifted);
}

/**
 * SVE 查表法（A 已预移位，无 alloc/shift 开销）
 * @param table  256x256 的 float 查找表
 * @param A_shifted  预左移 8 位的 A 矩阵 (uint32_t, N×L)
 * @param B_T    S*L 的 FP8 (uint8) 矩阵 (已转置)
 * @param C      N*S 的 float 结果矩阵
 *
 * A_shifted 由调用方一次性预处理，kernel 内部不再 alloc/free。
 */
void matmul_fp8_lookup_sve_preshift(
    const float *table, const uint32_t *A_shifted, const uint8_t *B_T,
    float *C, int N, int S, int L)
{
    for (int i = 0; i < N; ++i) {
        const uint32_t *rowA_sh = A_shifted + i * L;
        for (int j = 0; j < S; ++j) {
            const uint8_t *rowB = B_T + j * L;
            svfloat32_t acc_v = svdup_n_f32(0.0f);
            int k = 0;
            svbool_t pg = svwhilelt_b32(k, L);
            while (svptest_any(svptrue_b32(), pg)) {
                svuint32_t va_sh = svld1_u32(pg, &rowA_sh[k]);
                svuint32_t vb = svld1ub_u32(pg, &rowB[k]);
                svuint32_t indices = svorr_u32_z(pg, va_sh, vb);
                svfloat32_t vals = svld1_gather_u32index_f32(pg, table, indices);
                acc_v = svadd_f32_z(pg, acc_v, vals);
                k += svcntw();
                pg = svwhilelt_b32(k, L);
            }
            C[i * S + j] = svaddv_f32(svptrue_b32(), acc_v);
        }
    }
}

/**
 * 优化版 preshift：A_shifted 行预取 + gather 预取 + 内循环展开
 */
void matmul_fp8_lookup_sve_preshift_opt(
    const float *table, const uint32_t *A_shifted, const uint8_t *B_T,
    float *C, int N, int S, int L)
{
    for (int i = 0; i < N; ++i) {
        const uint32_t *rowA_sh = A_shifted + i * L;

        // Prefetch A_shifted 整行（2KB）进 L1
        for (int off = 0; off < L; off += 64 / sizeof(uint32_t)) {
            __builtin_prefetch(&rowA_sh[off], 0, 3);
        }

        for (int j = 0; j < S; ++j) {
            const uint8_t *rowB = B_T + j * L;
            svfloat32_t acc_v = svdup_n_f32(0.0f);

            // 主循环：2× 展开减少 whilelt/ptest 开销
            int k = 0;
            int step = svcntw() * 2;
            while (k + step <= L) {
                // --- 迭代 1 ---
                svbool_t pg = svptrue_b32();
                svuint32_t va_sh  = svld1_u32(pg, &rowA_sh[k]);
                svuint32_t vb     = svld1ub_u32(pg, &rowB[k]);
                svuint32_t idx    = svorr_u32_z(pg, va_sh, vb);
                svfloat32_t vals  = svld1_gather_u32index_f32(pg, table, idx);
                acc_v = svadd_f32_z(pg, acc_v, vals);

                // --- 迭代 2 ---
                va_sh  = svld1_u32(pg, &rowA_sh[k + svcntw()]);
                vb     = svld1ub_u32(pg, &rowB[k + svcntw()]);
                idx    = svorr_u32_z(pg, va_sh, vb);
                vals   = svld1_gather_u32index_f32(pg, table, idx);
                acc_v  = svadd_f32_z(pg, acc_v, vals);

                k += step;
            }
            // 尾部剩余迭代
            svbool_t pg = svwhilelt_b32(k, L);
            while (svptest_any(svptrue_b32(), pg)) {
                svuint32_t va_sh = svld1_u32(pg, &rowA_sh[k]);
                svuint32_t vb = svld1ub_u32(pg, &rowB[k]);
                svuint32_t indices = svorr_u32_z(pg, va_sh, vb);
                svfloat32_t vals = svld1_gather_u32index_f32(pg, table, indices);
                acc_v = svadd_f32_z(pg, acc_v, vals);
                k += svcntw();
                pg = svwhilelt_b32(k, L);
            }
            C[i * S + j] = svaddv_f32(svptrue_b32(), acc_v);
        }
    }
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
                svfloat16_t va = svld1_f16(pg, &rowA[k]);
                svfloat16_t vb = svld1_f16(pg, &rowB[k]);

                // 乘加运算
                acc_v = svmla_f16_z(pg, acc_v, va, vb);

                k += svcnth();
                pg = svwhilelt_b16(k, L);
            }
            C[i * S + j] = (float)svaddv_f16(svptrue_b16(), acc_v);
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
void matmul_int8_scalar(const int8_t *A, const int8_t *B_T, int32_t *C, int N, int S, int L)
{
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < S; ++j) {
            int32_t sum = 0;
            const int8_t *rowA = A + i * L;
            const int8_t *rowB = B_T + j * L;

            for (int k = 0; k < L; ++k) {
                sum += (int32_t)rowA[k] * (int32_t)rowB[k];
            }
            C[i * S + j] = sum;
        }
    }
}
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
                // 1. 加载 int8 向量
                svint8_t va = svld1_s8(pg, &rowA[k]);
                svint8_t vb = svld1_s8(pg, &rowB[k]);

                // 2. 核心：点积指令
                acc_v = svdot_s32(acc_v, va, vb);

                // 3. 步进：svcntb() 返回当前字节通道数
                k += svcntb();
                pg = svwhilelt_b8(k, L);
            }
            C[i * S + j] = svaddv_s32(svptrue_b32(), acc_v);
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
                svbool_t pg = svwhilelt_b8(k, L);
                svint8_t va0 = svld1_s8(pg, &rA0[k]), va1 = svld1_s8(pg, &rA1[k]);
                svint8_t vb0 = svld1_s8(pg, &rB0[k]), vb1 = svld1_s8(pg, &rB1[k]);

                acc00 = svmmla_s32(acc00, va0, vb0);
                acc01 = svmmla_s32(acc01, va0, vb1);
                acc10 = svmmla_s32(acc10, va1, vb0);
                acc11 = svmmla_s32(acc11, va1, vb1);
            }
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
            }
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
            }
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

// ─── 多核并行版：OpenMP 行拆分，处理 [i_start, i_end) 行 ───
static void matmul_fp8_rows(const float *table, const uint32_t *A_shifted,
                             const uint8_t *B_T, float *C,
                             int N, int S, int L, int i_start, int i_end)
{
    for (int i = i_start; i < i_end; ++i) {
        const uint32_t *rowA_sh = A_shifted + i * L;
        for (int off = 0; off < L; off += 64 / sizeof(uint32_t))
            __builtin_prefetch(&rowA_sh[off], 0, 3);

        for (int j = 0; j < S; ++j) {
            const uint8_t *rowB = B_T + j * L;
            svfloat32_t acc_v = svdup_n_f32(0.0f);
            int k = 0;
            int step = svcntw() * 2;
            while (k + step <= L) {
                svbool_t pg = svptrue_b32();
                svuint32_t va = svld1_u32(pg, &rowA_sh[k]);
                svuint32_t vb = svld1ub_u32(pg, &rowB[k]);
                svfloat32_t vals = svld1_gather_u32index_f32(pg, table, svorr_u32_z(pg, va, vb));
                acc_v = svadd_f32_z(pg, acc_v, vals);
                va = svld1_u32(pg, &rowA_sh[k + svcntw()]);
                vb = svld1ub_u32(pg, &rowB[k + svcntw()]);
                vals = svld1_gather_u32index_f32(pg, table, svorr_u32_z(pg, va, vb));
                acc_v = svadd_f32_z(pg, acc_v, vals);
                k += step;
            }
            svbool_t pg = svwhilelt_b32(k, L);
            while (svptest_any(svptrue_b32(), pg)) {
                svbool_t pg2 = svwhilelt_b32(k, L);
                svuint32_t va = svld1_u32(pg2, &rowA_sh[k]);
                svuint32_t vb = svld1ub_u32(pg2, &rowB[k]);
                svfloat32_t vals = svld1_gather_u32index_f32(pg2, table, svorr_u32_z(pg2, va, vb));
                acc_v = svadd_f32_z(pg2, acc_v, vals);
                k += svcntw();
                pg = svwhilelt_b32(k, L);
            }
            C[i * S + j] = svaddv_f32(svptrue_b32(), acc_v);
        }
    }
}

static void matmul_fp8_multicore(const float *table, const uint32_t *A_shifted,
                                  const uint8_t *B_T, float *C,
                                  int N, int S, int L, int num_threads)
{
    #pragma omp parallel num_threads(num_threads)
    {
        int t = omp_get_thread_num();
        int nt = omp_get_num_threads();
        int rows_per = (N + nt - 1) / nt;
        int i_start = t * rows_per;
        int i_end = std::min(i_start + rows_per, N);
        matmul_fp8_rows(table, A_shifted, B_T, C, N, S, L, i_start, i_end);
    }
}

int main(int argc, char **argv)
{
    // 分配内存
    std::vector<uint8_t> a_fp8(g_N * g_L), b_fp8(g_S * g_L);
    std::vector<float> lut(256 * 256);
    std::vector<__fp16> a_fp16(g_N * g_L), b_fp16(g_S * g_L);
    std::vector<float> res(g_N * g_S);

    std::vector<int8_t> a_i8(g_N * g_L), b_i8(g_S * g_L);
    std::vector<int32_t> res_i32(g_N * g_S);

    // 预分配 A_shifted 缓冲区（供 preshift 实验使用）
    uint32_t *A_shifted = (uint32_t *)aligned_alloc(64, (size_t)g_N * g_L * sizeof(uint32_t));

    // 读取 5 个文件
    load_bin("./input/matrix_a_8.bin", a_fp8.data(), g_N * g_L);
    load_bin("./input/matrix_b_8.bin", b_fp8.data(), g_S * g_L);
    load_bin("./input/look_up_table_fp32.bin", lut.data(), 256 * 256);
    load_bin("./input/matrix_a_16.bin", a_fp16.data(), g_N * g_L);
    load_bin("./input/matrix_b_16.bin", b_fp16.data(), g_S * g_L);
    load_bin("./input/matrix_a_i8.bin", a_i8.data(), g_N * g_L);
    load_bin("./input/matrix_b_i8.bin", b_i8.data(), g_S * g_L);

    // 初始化 A_shifted（文件数据已加载）
    for (int i = 0; i < g_N * g_L; ++i) {
        A_shifted[i] = (uint32_t)a_fp8[i] << 8;
    }

    // ─── 命令行参数解析 ───
    bool run_all = true;
    bool run_fp8_scalar = false, run_fp8_sve = false, run_fp8_opt = false;
    bool run_fp8_preshift = false, run_fp8_preshift_opt = false;
    bool run_fp16 = false, run_i8_scalar = false, run_i8_sve = false, run_i8mm = false;
    bool run_multicore = false;

    if (argc > 1) {
        run_all = false;
        for (int a = 1; a < argc; ++a) {
            std::string arg = argv[a];
            if (arg == "all")              run_all = true;
            else if (arg == "fp8_scalar")  run_fp8_scalar = true;
            else if (arg == "fp8_sve")     run_fp8_sve = true;
            else if (arg == "fp8_opt")     run_fp8_opt = true;
            else if (arg == "fp8_preshift") run_fp8_preshift = true;
            else if (arg == "fp8_preshift_opt" || arg == "preshift_opt") run_fp8_preshift_opt = true;
            else if (arg == "fp16")         run_fp16 = true;
            else if (arg == "i8_scalar")    run_i8_scalar = true;
            else if (arg == "i8_sve")       run_i8_sve = true;
            else if (arg == "i8mm")         run_i8mm = true;
            else if (arg == "fp8")          run_fp8_scalar = run_fp8_sve = run_fp8_opt = run_fp8_preshift = run_fp8_preshift_opt = true;
            else if (arg == "i8")           run_i8_scalar = run_i8_sve = run_i8mm = true;
            else if (arg == "multicore")    run_multicore = true;
            else if (arg == "-h" || arg == "--help") {
                printf("Usage: %s [kernels...]\n", argv[0]);
                printf("  all               Run all kernels (default)\n");
                printf("  fp8_scalar        FP8 scalar lookup\n");
                printf("  fp8_sve           FP8 SVE lookup\n");
                printf("  fp8_opt           FP8 SVE optimized\n");
                printf("  fp8_preshift      FP8 SVE preshift\n");
                printf("  fp8_preshift_opt  FP8 SVE preshift_opt\n");
                printf("  fp16              FP16 SVE fmla\n");
                printf("  i8_scalar         INT8 scalar\n");
                printf("  i8_sve            INT8 SVE sdot\n");
                printf("  i8mm              INT8 I8MM smmla\n");
                printf("  fp8               All FP8 lookup kernels\n");
                printf("  i8                All INT8 kernels\n");
                printf("  multicore         FP8 SVE 多核扩展实验 (1/2/4/8/16/32/64 核)\n");
                printf("  -h, --help        Show this help\n");
                return 0;
            }
            else {
                printf("Unknown arg: %s (use -h for help)\n", argv[a]);
                return 1;
            }
        }
    }

    if (run_all) {
        run_fp8_scalar = run_fp8_sve = run_fp8_opt = run_fp8_preshift = run_fp8_preshift_opt = true;
        run_fp16 = true;
        run_i8_scalar = run_i8_sve = run_i8mm = true;
    }

    // 预热
    int loopCnt = 10000;
    int warmup = 100;
    std::cout << "Warming up (" << warmup << " iterations per kernel)..." << std::endl;
    for (int i = 0; i < warmup; ++i) {
        if (run_fp8_scalar)      matmul_fp8_lookup_scalar(lut.data(), a_fp8.data(), b_fp8.data(), res.data(), g_N, g_S, g_L);
        if (run_fp8_sve)         matmul_fp8_lookup_sve(lut.data(), a_fp8.data(), b_fp8.data(), res.data(), g_N, g_S, g_L);
        if (run_fp8_opt)         matmul_fp8_lookup_sve_optimized(lut.data(), a_fp8.data(), b_fp8.data(), res.data(), g_N, g_S, g_L);
        if (run_fp8_preshift)    matmul_fp8_lookup_sve_preshift(lut.data(), A_shifted, b_fp8.data(), res.data(), g_N, g_S, g_L);
        if (run_fp8_preshift_opt) matmul_fp8_lookup_sve_preshift_opt(lut.data(), A_shifted, b_fp8.data(), res.data(), g_N, g_S, g_L);
        if (run_fp16)            matmul_sve_fp16(a_fp16.data(), b_fp16.data(), res.data(), g_N, g_S, g_L);
        if (run_i8_scalar)       matmul_int8_scalar(a_i8.data(), b_i8.data(), res_i32.data(), g_N, g_S, g_L);
        if (run_i8_sve)          matmul_int8_sve(a_i8.data(), b_i8.data(), res_i32.data(), g_N, g_S, g_L);
        if (run_i8mm)            matmul_int8_i8mm_complete(a_i8.data(), b_i8.data(), res_i32.data(), g_N, g_S, g_L);
    }
    {  // 查表计算
        bool any_fp8 = run_fp8_scalar || run_fp8_sve || run_fp8_opt || run_fp8_preshift || run_fp8_preshift_opt;
        if (any_fp8) std::cout << "\nComputing FP8 LUT MatMul..." << std::endl;
        std::vector<float> times(loopCnt);
        if (run_fp8_scalar) {
            for (int i = 0; i < loopCnt; ++i) {
                TimoPoint tpBegin = std::chrono::high_resolution_clock::now();
                matmul_fp8_lookup_scalar(lut.data(), a_fp8.data(), b_fp8.data(), res.data(), g_N, g_S, g_L);
                TimoPoint tpEnd = std::chrono::high_resolution_clock::now();
                times[i] = (tpEnd - tpBegin).count() / 1000;
            }
            std::cout << "FP8 lookup scalar dura = " << Average(times) << " us" << std::endl;
        }
        if (run_fp8_sve) {
            for (int i = 0; i < loopCnt; ++i) {
                TimoPoint tpBegin = std::chrono::high_resolution_clock::now();
                matmul_fp8_lookup_sve(lut.data(), a_fp8.data(), b_fp8.data(), res.data(), g_N, g_S, g_L);
                TimoPoint tpEnd = std::chrono::high_resolution_clock::now();
                times[i] = (tpEnd - tpBegin).count() / 1000;
            }
            std::cout << "FP8 lookup sve dura = " << Average(times) << " us" << std::endl;
        }
        if (run_fp8_opt) {
            for (int i = 0; i < loopCnt; ++i) {
                TimoPoint tpBegin = std::chrono::high_resolution_clock::now();
                matmul_fp8_lookup_sve_optimized(lut.data(), a_fp8.data(), b_fp8.data(), res.data(), g_N, g_S, g_L);
                TimoPoint tpEnd = std::chrono::high_resolution_clock::now();
                times[i] = (tpEnd - tpBegin).count() / 1000;
            }
            std::cout << "FP8 lookup sve opt dura = " << Average(times) << " us" << std::endl;
        }

        if (run_fp8_preshift || run_fp8_preshift_opt) {
            // 预移位 A（一次性预处理，单独计时）
            TimoPoint tpPreshiftBegin = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < g_N * g_L; ++i) A_shifted[i] = (uint32_t)a_fp8[i] << 8;
            volatile uint32_t sink = 0;
            for (int i = 0; i < g_N * g_L; i += 64 / sizeof(uint32_t)) sink += A_shifted[i];
            TimoPoint tpPreshiftEnd = std::chrono::high_resolution_clock::now();
            double preshift_us = (tpPreshiftEnd - tpPreshiftBegin).count() / 1000.0;
            std::cout << "A_shifted preprocess dura = " << preshift_us << " us (one-time)" << std::endl;
        }
        if (run_fp8_preshift) {
            for (int i = 0; i < loopCnt; ++i) {
                TimoPoint tpBegin = std::chrono::high_resolution_clock::now();
                matmul_fp8_lookup_sve_preshift(lut.data(), A_shifted, b_fp8.data(), res.data(), g_N, g_S, g_L);
                TimoPoint tpEnd = std::chrono::high_resolution_clock::now();
                times[i] = (tpEnd - tpBegin).count() / 1000;
            }
            std::cout << "FP8 lookup sve preshift dura = " << Average(times) << " us" << std::endl;
        }
        if (run_fp8_preshift_opt) {
            for (int i = 0; i < loopCnt; ++i) {
                TimoPoint tpBegin = std::chrono::high_resolution_clock::now();
                matmul_fp8_lookup_sve_preshift_opt(lut.data(), A_shifted, b_fp8.data(), res.data(), g_N, g_S, g_L);
                TimoPoint tpEnd = std::chrono::high_resolution_clock::now();
                times[i] = (tpEnd - tpBegin).count() / 1000;
            }
            std::cout << "FP8 lookup sve preshift_opt dura = " << Average(times) << " us" << std::endl;
        }
    }
    if (run_fp16) {  // FP16 指令加速计算
        std::cout << "Computing FP16 sve MatMul..." << std::endl;
        TimoPoint t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < loopCnt; ++i) matmul_sve_fp16(a_fp16.data(), b_fp16.data(), res.data(), g_N, g_S, g_L);
        TimoPoint t1 = std::chrono::high_resolution_clock::now();
        std::cout << "FP16 sve mat mul dura = " << (t1 - t0).count() / loopCnt / 1000.0 << " us" << std::endl;
    }
    if (run_i8_scalar || run_i8_sve || run_i8mm) {  // i8矩阵乘法运算
        std::cout << "Computing i8 MatMul..." << std::endl;
        if (run_i8_scalar) {
            TimoPoint t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < loopCnt; ++i) matmul_int8_scalar(a_i8.data(), b_i8.data(), res_i32.data(), g_N, g_S, g_L);
            TimoPoint t1 = std::chrono::high_resolution_clock::now();
            std::cout << "i8 scalar mat mul dura = " << (t1 - t0).count() / loopCnt / 1000.0 << " us" << std::endl;
        }
        if (run_i8_sve) {
            TimoPoint t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < loopCnt; ++i) matmul_int8_sve(a_i8.data(), b_i8.data(), res_i32.data(), g_N, g_S, g_L);
            TimoPoint t1 = std::chrono::high_resolution_clock::now();
            std::cout << "i8 sve mat mul dura = " << (t1 - t0).count() / loopCnt / 1000.0 << " us" << std::endl;
        }
        if (run_i8mm) {
            TimoPoint t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < loopCnt; ++i) matmul_int8_i8mm_complete(a_i8.data(), b_i8.data(), res_i32.data(), g_N, g_S, g_L);
            TimoPoint t1 = std::chrono::high_resolution_clock::now();
            std::cout << "i8 mm mat mul dura = " << (t1 - t0).count() / loopCnt / 1000.0 << " us" << std::endl;
        }
    }
    // ─── Cache miss 率测量（perf_event_open） ───
    {
        bool any_cache = run_fp8_scalar || run_fp8_sve || run_fp8_opt || run_fp8_preshift || run_fp8_preshift_opt
                      || run_fp16 || run_i8_scalar || run_i8_sve || run_i8mm;
        if (any_cache) {
            std::cout << "\n--- Cache Miss Rates (L1/L2) ---\n";
            std::cout << std::left << std::setw(32) << "Kernel"
                      << std::setw(14) << "L1-miss%"
                      << std::setw(14) << "L2-miss%" << "\n";
            std::cout << std::string(60, '-') << "\n";
        }

        auto report = [&](const std::string &name, const std::function<void()> &fn) {
            CacheCounters cc = measure_cache(fn, 10, 100);
            std::cout << std::left << std::setw(32) << name;
            if (cc.ok && cc.l1_access > 0 && cc.l2_access > 0) {
                double l1 = 100.0 * cc.l1_refill / cc.l1_access;
                double l2 = 100.0 * cc.l2_refill / cc.l2_access;
                std::cout << std::fixed << std::setprecision(2)
                          << std::setw(14) << l1
                          << std::setw(14) << l2 << "%";
            } else if (!cc.ok) {
                std::cout << std::setw(14) << "N/A" << std::setw(14) << "N/A (non-ARM?)";
            } else {
                std::cout << std::setw(14) << "0?" << std::setw(14) << "0?";
            }
            std::cout << "\n";
        };

        if (run_fp8_scalar)      report("FP8 标量查表",     [&](){ matmul_fp8_lookup_scalar(lut.data(), a_fp8.data(), b_fp8.data(), res.data(), g_N, g_S, g_L); });
        if (run_fp8_sve)         report("FP8 SVE 查表",     [&](){ matmul_fp8_lookup_sve(lut.data(), a_fp8.data(), b_fp8.data(), res.data(), g_N, g_S, g_L); });
        if (run_fp8_opt)         report("FP8 SVE 优化版",   [&](){ matmul_fp8_lookup_sve_optimized(lut.data(), a_fp8.data(), b_fp8.data(), res.data(), g_N, g_S, g_L); });
        if (run_fp8_preshift)    report("FP8 SVE preshift", [&](){ matmul_fp8_lookup_sve_preshift(lut.data(), A_shifted, b_fp8.data(), res.data(), g_N, g_S, g_L); });
        if (run_fp8_preshift_opt) report("FP8 SVE preshift_opt", [&](){ matmul_fp8_lookup_sve_preshift_opt(lut.data(), A_shifted, b_fp8.data(), res.data(), g_N, g_S, g_L); });
        if (run_fp16)            report("FP16 SVE fmla",    [&](){ matmul_sve_fp16(a_fp16.data(), b_fp16.data(), res.data(), g_N, g_S, g_L); });
        if (run_i8_scalar)       report("INT8 标量", [&](){ matmul_int8_scalar(a_i8.data(), b_i8.data(), res_i32.data(), g_N, g_S, g_L); });
        if (run_i8_sve)          report("INT8 SVE sdot",    [&](){ matmul_int8_sve(a_i8.data(), b_i8.data(), res_i32.data(), g_N, g_S, g_L); });
        if (run_i8mm)            report("INT8 I8MM smmla",  [&](){ matmul_int8_i8mm_complete(a_i8.data(), b_i8.data(), res_i32.data(), g_N, g_S, g_L); });
        if (any_cache) std::cout << "\n";
    }

    // ─── 多核 FP8 SVE 扩展实验 ───
    if (run_multicore) {
        const int max_cores = omp_get_max_threads();
        int thread_counts[] = {1, 2, 4, 8, 16, 32, 64};
        int num_configs = 0;
        for (int c : thread_counts) {
            if (c <= max_cores) num_configs++;
            else break;
        }

        std::cout << "\n══════════════════════════════════════════════════════════\n";
        std::cout << "多核 FP8 SVE preshift_opt 性能\n";
        std::cout << "LUT = 256×256 float (256 KiB, 全部 L2 resident)\n";
        std::cout << "N=" << g_N << " S=" << g_S << " L=" << g_L;
        std::cout << "  核数上限=" << max_cores << "\n";
        std::cout << "══════════════════════════════════════════════════════════\n";
        std::cout << std::left << std::setw(8) << "Cores"
                  << std::setw(16) << "Time(μs)"
                  << std::setw(14) << "GOP/s"
                  << std::setw(12) << "Speedup"
                  << std::setw(14) << "L1-miss%"
                  << std::setw(14) << "L2-miss%" << "\n";
        std::cout << std::string(78, '-') << "\n";

        int mc_warmup = 10, mc_repeat = 100;
        double base_time = 0;

        for (int ti = 0; ti < num_configs; ++ti) {
            int tc = thread_counts[ti];
            auto fn = [&]() { matmul_fp8_multicore(lut.data(), A_shifted, b_fp8.data(), res.data(), g_N, g_S, g_L, tc); };

            // Warmup
            for (int i = 0; i < mc_warmup; ++i) fn();

            // Timing
            TimoPoint t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < mc_repeat; ++i) fn();
            TimoPoint t1 = std::chrono::high_resolution_clock::now();
            double avg_us = (t1 - t0).count() / (double)mc_repeat / 1000.0;

            double gops = 1.0 * g_N * g_S * g_L / avg_us / 1000.0;
            double speedup = (tc == 1) ? 1.0 : base_time / avg_us;
            if (tc == 1) base_time = avg_us;

            // Cache miss（进程级 aggregate）
            PerfCounter c_l1a(ArmPmu::L1D_CACHE);
            PerfCounter c_l1r(ArmPmu::L1D_CACHE_REFILL);
            PerfCounter c_l2a(ArmPmu::L2D_CACHE);
            PerfCounter c_l2r(ArmPmu::L2D_CACHE_REFILL);
            bool cache_ok = c_l1a.ok() && c_l1r.ok() && c_l2a.ok() && c_l2r.ok();
            double l1_mr = 0, l2_mr = 0;
            if (cache_ok) {
                c_l1a.reset(); c_l1r.reset(); c_l2a.reset(); c_l2r.reset();
                c_l1a.enable(); c_l1r.enable(); c_l2a.enable(); c_l2r.enable();
                int cache_rep = 20;
                for (int i = 0; i < cache_rep; ++i) fn();
                c_l1a.disable(); c_l1r.disable(); c_l2a.disable(); c_l2r.disable();
                uint64_t l1a_v = c_l1a.read(), l1r_v = c_l1r.read();
                uint64_t l2a_v = c_l2a.read(), l2r_v = c_l2r.read();
                if (l1a_v && l2a_v) {
                    l1_mr = 100.0 * l1r_v / l1a_v;
                    l2_mr = 100.0 * l2r_v / l2a_v;
                } else { cache_ok = false; }
            }

            std::cout << std::left << std::setw(8) << tc
                      << std::fixed << std::setprecision(1) << std::setw(16) << avg_us
                      << std::setprecision(2) << std::setw(14) << gops
                      << std::setprecision(2) << std::setw(12) << speedup;
            if (cache_ok)
                std::cout << std::setprecision(2) << std::setw(14) << l1_mr
                          << std::setprecision(2) << std::setw(14) << l2_mr << "%";
            else
                std::cout << std::setw(14) << "N/A" << std::setw(14) << "N/A";
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    free(A_shifted);
    return 0;
}