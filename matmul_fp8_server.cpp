//============================================================================
// FP8 查表法矩阵乘法 - 服务器性能测试 (ARM SVE + NUMA)
// 目标平台: HiSilicon Kunpeng (鲲鹏), EulerOS, 80核×4路, SVE
// 编译: g++ -O3 -fopenmp -march=armv8.2-a+sve -o fp8_server matmul_fp8_server.cpp -std=c++17 -lnuma
// 依赖: sudo dnf install -y numactl-devel
// 运行: OMP_PLACES=cores OMP_PROC_BIND=close ./fp8_server
//============================================================================
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <arm_sve.h>
#include <omp.h>
#include <sstream>
#include <functional>
#include <algorithm>
#include <unistd.h>
#include <numa.h>

using TimePoint = std::chrono::high_resolution_clock::time_point;
using namespace std::chrono;

// ====================== 系统参数 ======================

static constexpr int L1D_KiB  = 64;     // 鲲鹏 L1d 64 KiB/核
static constexpr int L2_KiB   = 1280;   // 鲲鹏 L2 1.25 MiB/核
static constexpr int L3_KiB   = 71680;  // 每 NUMA node 70 MiB

static constexpr int LUT_SIZE = 256 * 256;
static constexpr int LUT_KiB  = 256;    // 256×256 float = 256 KiB

// ====================== 数据生成 ======================

void gen_lut(float *lut) {
    for (int i = 0; i < 256; i++)
        for (int j = 0; j < 256; j++)
            lut[i * 256 + j] = sinf(i * 0.1f) * cosf(j * 0.1f);
}

template <typename T>
void fill_random(T *data, size_t n) {
    for (size_t i = 0; i < n; i++) data[i] = static_cast<T>(rand() % 256);
}

// ====================== Cache 分析工具 ======================

int calc_tile_workset_kib(int Ni, int Sj, int L) {
    return (Ni * L * 1 + Sj * L * 1 + Ni * Sj * 4 + LUT_KiB * 1024) / 1024;
}

const char* cache_level(int workset_kib, int L1d, int L2) {
    if (workset_kib <= L1d) return "L1";
    if (workset_kib <= L2)  return "L2";
    return "L3+";
}

double format_size(int kib, const char*& unit) {
    if (kib < 1024) { unit = "KiB"; return kib; }
    unit = "MiB"; return kib / 1024.0;
}

// ====================== 计算核心 ======================

// ---- 标量版（串行基线） ----
void lookup_scalar(const float *table, const uint8_t *A, const uint8_t *B_T,
                   float *C, int N, int S, int L) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < S; ++j) {
            float sum = 0.0f;
            const uint8_t *rA = A + i * L, *rB = B_T + j * L;
            for (int k = 0; k < L; ++k)
                sum += table[(rA[k] << 8) | rB[k]];
            C[i * S + j] = sum;
        }
}

// ---- SVE gather 版（串行） ----
void lookup_sve(const float *table, const uint8_t *A, const uint8_t *B_T,
                float *C, int N, int S, int L) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < S; ++j) {
            svfloat32_t acc = svdup_n_f32(0.0f);
            const uint8_t *rA = A + i * L, *rB = B_T + j * L;
            int k = 0;
            svbool_t pg = svwhilelt_b32(k, L);
            while (svptest_any(svptrue_b32(), pg)) {
                svuint32_t idxA = svld1ub_u32(pg, &rA[k]);
                svuint32_t idxB = svld1ub_u32(pg, &rB[k]);
                svuint32_t idx = svorr_u32_z(pg, svlsl_n_u32_z(pg, idxA, 8), idxB);
                acc = svadd_f32_z(pg, acc, svld1_gather_u32index_f32(pg, table, idx));
                k += svcntw();
                pg = svwhilelt_b32(k, L);
            }
            C[i * S + j] = svaddv_f32(svptrue_b32(), acc);
        }
}

// ---- SVE + OpenMP ----
void lookup_sve_omp(const float *table, const uint8_t *A, const uint8_t *B_T,
                    float *C, int N, int S, int L) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < S; ++j) {
            svfloat32_t acc = svdup_n_f32(0.0f);
            const uint8_t *rA = A + i * L, *rB = B_T + j * L;
            int k = 0;
            svbool_t pg = svwhilelt_b32(k, L);
            while (svptest_any(svptrue_b32(), pg)) {
                svuint32_t idxA = svld1ub_u32(pg, &rA[k]);
                svuint32_t idxB = svld1ub_u32(pg, &rB[k]);
                svuint32_t idx = svorr_u32_z(pg, svlsl_n_u32_z(pg, idxA, 8), idxB);
                acc = svadd_f32_z(pg, acc, svld1_gather_u32index_f32(pg, table, idx));
                k += svcntw();
                pg = svwhilelt_b32(k, L);
            }
            C[i * S + j] = svaddv_f32(svptrue_b32(), acc);
        }
}

// ---- SVE + Tiling + OpenMP ----
void lookup_sve_tiled_omp(const float *table, const uint8_t *A, const uint8_t *B_T,
                           float *C, int N, int S, int L, int Ni, int Sj) {
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ii = 0; ii < N; ii += Ni) {
        for (int jj = 0; jj < S; jj += Sj) {
            int i_end = std::min(ii + Ni, N);
            int j_end = std::min(jj + Sj, S);
            for (int i = ii; i < i_end; ++i)
                for (int j = jj; j < j_end; ++j) {
                    svfloat32_t acc = svdup_n_f32(0.0f);
                    const uint8_t *rA = A + i * L, *rB = B_T + j * L;
                    int k = 0;
                    svbool_t pg = svwhilelt_b32(k, L);
                    while (svptest_any(svptrue_b32(), pg)) {
                        svuint32_t idxA = svld1ub_u32(pg, &rA[k]);
                        svuint32_t idxB = svld1ub_u32(pg, &rB[k]);
                        svuint32_t idx = svorr_u32_z(pg, svlsl_n_u32_z(pg, idxA, 8), idxB);
                        acc = svadd_f32_z(pg, acc, svld1_gather_u32index_f32(pg, table, idx));
                        k += svcntw();
                        pg = svwhilelt_b32(k, L);
                    }
                    C[i * S + j] = svaddv_f32(svptrue_b32(), acc);
                }
        }
    }
}

// ====================== NUMA 工具 ======================

/** 打印 NUMA 拓扑 */
void print_numa_info() {
    if (numa_available() < 0) {
        std::cout << "  NUMA: not available\n";
        return;
    }
    int max_node = numa_max_node();
    std::cout << "  NUMA nodes: " << (max_node + 1) << "\n";
    for (int n = 0; n <= max_node; ++n) {
        if (numa_bitmask_isbitset(numa_all_nodes_ptr, n)) {
            struct bitmask *cpus = numa_allocate_cpumask();
            if (numa_node_to_cpus(n, cpus) == 0) {
                int count = 0;
                for (int i = 0; i < numa_num_configured_cpus(); i++)
                    if (numa_bitmask_isbitset(cpus, i)) count++;
                std::cout << "    Node " << n << ": " << count << " CPUs\n";
            }
            numa_free_cpumask(cpus);
        }
    }
}

/** 获取指定 NUMA node 上所有 CPU ID 列表 */
std::vector<int> cpus_on_node(int node) {
    std::vector<int> result;
    if (numa_available() < 0) return result;
    struct bitmask *mask = numa_allocate_cpumask();
    if (numa_node_to_cpus(node, mask) == 0) {
        for (int i = 0; i < numa_num_configured_cpus(); i++) {
            if (numa_bitmask_isbitset(mask, i))
                result.push_back(i);
        }
    }
    numa_free_cpumask(mask);
    return result;
}

/** 获取指定 NUMA node 的第一个 CPU ID */
int first_cpu_on_node(int node) {
    auto cpus = cpus_on_node(node);
    return cpus.empty() ? 0 : cpus[0];
}

/** 在指定 CPU 集合上运行 */
void pin_to_cpu_list(const std::vector<int>& cpus) {
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c : cpus) CPU_SET(c, &set);
    sched_setaffinity(0, sizeof(set), &set);
}

// ====================== 实验框架 ======================

/** 计时器：单次运行（含 warmup + 多次取最优） */
double bench(std::function<void()> fn, int warmup = 2, int samples = 5) {
    for (int t = 0; t < warmup; ++t) fn();
    double best = 1e18;
    for (int t = 0; t < samples; ++t) {
        auto start = high_resolution_clock::now();
        fn();
        auto end = high_resolution_clock::now();
        double us = duration_cast<nanoseconds>(end - start).count() / 1000.0;
        if (us < best) best = us;
    }
    return best;
}

/** 检查正确性 */
bool verify(const float *ref, const float *result, int n, float tol = 1e-3f) {
    for (int i = 0; i < n; i++)
        if (fabsf(ref[i] - result[i]) > tol) return false;
    return true;
}

double calc_gops(int N, int S, int L, double us) {
    return double(N) * double(S) * double(L) / us / 1e3;
}

// ====================== 实验1：SVE 加速比 ======================

void exp1_sve_speedup(const float *table, const uint8_t *A, const uint8_t *B_T,
                      int N, int S, int L) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "实验1: SVE 向量化加速比 (单核)\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "SVE 向量宽度: svcntw() = " << svcntw() << " floats, svcntb() = " << svcntb() << " bytes\n";
    std::cout << "矩阵: " << N << "×" << L << " * " << S << "×" << L << "^T\n";
    std::cout << "工作集: " << calc_tile_workset_kib(N, S, L) << " KiB\n\n";

    std::vector<float> C1(N * S), C2(N * S);

    omp_set_num_threads(1);
    double t_scalar = bench([&]() {
        lookup_scalar(table, A, B_T, C1.data(), N, S, L);
    });
    double t_sve = bench([&]() {
        lookup_sve(table, A, B_T, C2.data(), N, S, L);
    });

    bool ok = verify(C1.data(), C2.data(), N * S);

    std::cout << std::left
              << std::setw(20) << "Version"
              << std::setw(16) << "Time(us)"
              << std::setw(14) << "GOP/s"
              << std::setw(14) << "Speedup"
              << std::setw(10) << "Valid?" << "\n"
              << std::string(70, '-') << "\n";

    std::ostringstream ss1, ss2;
    ss1 << std::fixed << std::setprecision(2) << (N * S * L / t_scalar / 1e3);
    ss2 << std::fixed << std::setprecision(2) << (N * S * L / t_sve / 1e3);

    std::cout << std::setw(20) << "Scalar (标量)"
              << std::setw(16) << std::fixed << std::setprecision(1) << t_scalar
              << std::setw(14) << ss1.str()
              << std::setw(14) << "1.00×"
              << std::setw(10) << "—" << "\n";
    std::cout << std::setw(20) << "SVE gather"
              << std::setw(16) << std::fixed << std::setprecision(1) << t_sve
              << std::setw(14) << ss2.str()
              << std::setw(14) << std::fixed << std::setprecision(2) << (t_scalar/t_sve) << "×"
              << std::setw(10) << (ok ? "OK" : "FAIL") << "\n\n";
}

// ====================== 实验2：Tile 尺寸扫描 ======================

void exp2_tile_sweep(const float *table, const uint8_t *A, const uint8_t *B_T,
                     float *ref, int N, int S, int L, int num_threads) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "实验2: Tile 尺寸扫描 (SVE + " << num_threads << " 线程)\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "矩阵: " << N << "×" << L << " * " << S << "×" << L << "^T\n";
    std::cout << "L1d = " << L1D_KiB << " KiB, L2 = " << L2_KiB << " KiB/核\n\n";

    std::cout << std::left
              << std::setw(8) << "Ni"
              << std::setw(8) << "Sj"
              << std::setw(18) << "TileWorkSet"
              << std::setw(10) << "CacheLvl"
              << std::setw(14) << "Time(us)"
              << std::setw(10) << "GOP/s"
              << std::setw(10) << "Valid?"
              << "\n" << std::string(75, '-') << "\n";

    int ni_vals[] = {1, 2, 4, 8, 16, 32, 64, 128};
    int sj_vals[] = {8, 16, 32, 64, 128, 256, 512};

    omp_set_num_threads(num_threads);
    std::vector<float> C(N * S);

    for (int ni : ni_vals) {
        for (int sj : sj_vals) {
            if (ni > N || sj > S) continue;

            int ws = calc_tile_workset_kib(ni, sj, L);
            const char *cl = cache_level(ws, L1D_KiB, L2_KiB);
            const char *unit; double val = format_size(ws, unit);

            double best = bench([&]() {
                memset(C.data(), 0, N * S * sizeof(float));
                lookup_sve_tiled_omp(table, A, B_T, C.data(), N, S, L, ni, sj);
            });

            bool ok = verify(ref, C.data(), N * S);
            double gops = calc_gops(N, S, L, best);

            std::cout << std::left
                      << std::setw(8) << ni
                      << std::setw(8) << sj
                      << std::setw(10) << std::fixed << std::setprecision(1) << val
                      << std::setw(4) << unit
                      << std::setw(4) << " "
                      << std::setw(10) << cl
                      << std::setw(14) << std::fixed << std::setprecision(1) << best
                      << std::setw(10) << std::fixed << std::setprecision(2) << gops
                      << std::setw(10) << (ok ? "OK" : "FAIL")
                      << "\n";
        }
    }
}

// ====================== 实验3：单 NUMA Node 内扩展 ======================

void exp3_numa_scaling(const float *table, const uint8_t *A, const uint8_t *B_T,
                        float *ref, int N, int S, int L, int numa_node,
                        std::vector<int> numa_cpus) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "实验3: 单 NUMA Node 内多核扩展 (SVE, Node " << numa_node << ")\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "NUMA Node " << numa_node << " CPUs: " << numa_cpus.size() << " 个\n";
    std::cout << "矩阵: " << N << "×" << L << " * " << S << "×" << L << "^T\n\n";

    // 列出要测试的线程数
    std::vector<int> t_counts;
    for (int t : {1, 2, 4, 8, 16, 32, 64})
        if (t <= (int)numa_cpus.size()) t_counts.push_back(t);
    if ((int)numa_cpus.size() > 64) t_counts.push_back(numa_cpus.size());

    std::cout << std::left
              << std::setw(10) << "Threads"
              << std::setw(14) << "Time(us)"
              << std::setw(12) << "Speedup"
              << std::setw(12) << "Efficiency"
              << std::setw(12) << "GOP/s"
              << std::setw(10) << "Valid?"
              << "\n" << std::string(70, '-') << "\n";

    std::vector<float> C(N * S);
    double t1 = 0;

    // 每个线程数：绑定到前 t 个 CPU
    for (int nt : t_counts) {
        // 将当前线程绑定到前 nt 个 CPU
        cpu_set_t set; CPU_ZERO(&set);
        for (int i = 0; i < nt; i++) CPU_SET(numa_cpus[i], &set);
        sched_setaffinity(0, sizeof(set), &set);

        omp_set_num_threads(nt);
        // 设置 OpenMP 线程绑定
        setenv("OMP_NUM_THREADS", std::to_string(nt).c_str(), 1);
        setenv("OMP_PLACES", "cores", 1);
        setenv("OMP_PROC_BIND", "close", 1);

        double best = bench([&]() {
            memset(C.data(), 0, N * S * sizeof(float));
            lookup_sve_omp(table, A, B_T, C.data(), N, S, L);
        });

        if (nt == 1) t1 = best;

        bool ok = verify(ref, C.data(), N * S);
        double gops = calc_gops(N, S, L, best);
        double speedup = t1 / best;
        double eff = speedup / nt * 100;

        std::ostringstream ss_sp, ss_eff, ss_go;
        ss_sp  << std::fixed << std::setprecision(2) << speedup << "×";
        ss_eff << std::fixed << std::setprecision(0) << eff << "%";
        ss_go  << std::fixed << std::setprecision(2) << gops;

        std::cout << std::left
                  << std::setw(10) << nt
                  << std::setw(14) << std::fixed << std::setprecision(1) << best
                  << std::setw(12) << ss_sp.str()
                  << std::setw(12) << ss_eff.str()
                  << std::setw(12) << ss_go.str()
                  << std::setw(10) << (ok ? "OK" : "FAIL")
                  << "\n";
    }
    std::cout << "\n";
}

// ====================== 实验4：跨 NUMA 扩展 ======================

void exp4_cross_numa(const float *table, int N, int S, int L) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "实验4: 跨 NUMA 扩展 (SVE + Tiling)\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "矩阵: " << N << "×" << L << " * " << S << "×" << L << "^T\n";
    std::cout << "Tile: 32×128 (工作集 352 KiB → L2)\n\n";

    // 分配 NUMA-local 内存
    std::vector<uint8_t> A, B_T;
    std::vector<float> LUT(LUT_SIZE), ref;
    gen_lut(LUT.data());

    // 取本地的数据
    A.resize(N * L); fill_random(A.data(), N * L);
    B_T.resize(S * L); fill_random(B_T.data(), S * L);
    ref.resize(N * S);

    // 单线程参考
    omp_set_num_threads(1);
    lookup_sve(table, A.data(), B_T.data(), ref.data(), N, S, L);

    // 跨 NUMA 配置 — 直接尝试硬编码的线程数，不依赖 omp_get_max_threads
    struct NumaConfig {
        const char* desc;
        int thread_count;
    };
    std::vector<NumaConfig> configs = {
        {"1", 1},
        {"80 (1 node)", 80},
        {"160 (2 nodes)", 160},
        {"240 (3 nodes)", 240},
        {"320 (4 nodes)", 320},
    };

    const int Ni = 32, Sj = 128;

    std::cout << std::left
              << std::setw(14) << "Threads"
              << std::setw(18) << "Config"
              << std::setw(14) << "Time(us)"
              << std::setw(12) << "GOP/s"
              << std::setw(12) << "Speedup"
              << std::setw(10) << "Valid?"
              << "\n" << std::string(75, '-') << "\n";

    double t1 = 0;
    for (auto &cfg : configs) {
        int nt = cfg.thread_count;
        omp_set_num_threads(nt);
        setenv("OMP_NUM_THREADS", std::to_string(nt).c_str(), 1);
        setenv("OMP_PLACES", "cores", 1);
        setenv("OMP_PROC_BIND", "spread", 1);  // spread 跨 NUMA

        std::vector<float> C(N * S);
        double best = bench([&]() {
            memset(C.data(), 0, N * S * sizeof(float));
            lookup_sve_tiled_omp(LUT.data(), A.data(), B_T.data(), C.data(), N, S, L, Ni, Sj);
        });

        if (nt == 1) t1 = best;
        bool ok = verify(ref.data(), C.data(), N * S);
        double gops = calc_gops(N, S, L, best);
        double sp = t1 / best;

        std::cout << std::left
                  << std::setw(14) << nt
                  << std::setw(18) << cfg.desc
                  << std::setw(14) << std::fixed << std::setprecision(1) << best
                  << std::setw(12) << std::fixed << std::setprecision(2) << gops
                  << std::setw(12) << std::fixed << std::setprecision(2) << sp << "×"
                  << std::setw(10) << (ok ? "OK" : "FAIL")
                  << "\n";
    }
    std::cout << "\n";
}

// ====================== 实验5：矩阵规模扩展 ======================

void exp5_matrix_scaling(const float *table) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "实验5: 矩阵规模扩展 × Tile 优化效果\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "L1d = 64 KiB/核, L2 = 1280 KiB/核, L3 = 70 MiB/NUMA node\n\n";

    struct { int N, S, L; const char* label; } scales[] = {
        {256,  1024, 512, "1×  (256×1024)"},
        {512,  2048, 512, "2×  (512×2048)"},
        {1024, 4096, 512, "4×  (1024×4096)"},
        {2048, 8192, 512, "8×  (2048×8192)"},
    };

    std::cout << std::left
              << std::setw(16) << "Scale"
              << std::setw(10) << "Threads"
              << std::setw(10) << "Tiling"
              << std::setw(16) << "WorkSet"
              << std::setw(10) << "Cache"
              << std::setw(14) << "Time(us)"
              << std::setw(10) << "GOP/s"
              << "\n" << std::string(80, '-') << "\n";

    for (auto &s : scales) {
        int N = s.N, S = s.S, L = s.L;

        std::vector<uint8_t> A(N * L), B_T(S * L);
        fill_random(A.data(), N * L);
        fill_random(B_T.data(), S * L);
        std::vector<float> ref(N * S);

        // 参考结果 (1 thread, scalar)
        omp_set_num_threads(1);
        lookup_sve(table, A.data(), B_T.data(), ref.data(), N, S, L);

        int ws = calc_tile_workset_kib(N, S, L);
        const char *cl = cache_level(ws, L1D_KiB, L2_KiB);
        // 多核下每核有效 L3 ~ 70M/80 = 0.875M (实际取决于竞争)
        const char *cl_multi = (ws <= L2_KiB) ? "L2" :
                              (ws <= 70000) ? "L3" : "DRAM";
        const char *unit; double val = format_size(ws, unit);

        // --- 1 thread, no tile ---
        {
            omp_set_num_threads(1);
            double best = bench([&]() {
                lookup_sve(table, A.data(), B_T.data(), ref.data(), N, S, L);
            });
            double gops = calc_gops(N, S, L, best);
            std::cout << std::left
                      << std::setw(16) << s.label
                      << std::setw(10) << 1
                      << std::setw(10) << "none"
                      << std::setw(8) << std::fixed << std::setprecision(1) << val
                      << std::setw(4) << unit
                      << std::setw(4) << " "
                      << std::setw(10) << cl
                      << std::setw(14) << std::fixed << std::setprecision(1) << best
                      << std::setw(10) << std::fixed << std::setprecision(2) << gops
                      << "\n";
        }

        // --- 80 threads (1 NUMA node), no tile ---
        {
            omp_set_num_threads(80);
            std::vector<float> C(N * S);
            double best = bench([&]() {
                memset(C.data(), 0, N * S * sizeof(float));
                lookup_sve_omp(table, A.data(), B_T.data(), C.data(), N, S, L);
            });
            bool ok = verify(ref.data(), C.data(), N * S);
            double gops = calc_gops(N, S, L, best);
            std::cout << std::left
                      << std::setw(16) << ""
                      << std::setw(10) << 80
                      << std::setw(10) << "none"
                      << std::setw(8) << std::fixed << std::setprecision(1) << val
                      << std::setw(4) << unit
                      << std::setw(4) << " "
                      << std::setw(10) << cl_multi
                      << std::setw(14) << std::fixed << std::setprecision(1) << best
                      << std::setw(10) << std::fixed << std::setprecision(2) << gops
                      << (ok ? "" : " FAIL")
                      << "\n";
        }

        // --- 80 threads, tiled (32×128) ---
        {
            int Ni = std::min(32, N), Sj = std::min(128, S);
            int ws_t = calc_tile_workset_kib(Ni, Sj, L);
            const char *cl_t = cache_level(ws_t, L1D_KiB, L2_KiB);
            const char *unit_t; double val_t = format_size(ws_t, unit_t);

            omp_set_num_threads(80);
            std::vector<float> C(N * S);
            double best = bench([&]() {
                memset(C.data(), 0, N * S * sizeof(float));
                lookup_sve_tiled_omp(table, A.data(), B_T.data(), C.data(), N, S, L, Ni, Sj);
            });
            bool ok = verify(ref.data(), C.data(), N * S);
            double gops = calc_gops(N, S, L, best);

            std::string tile_label = std::to_string(Ni) + "×" + std::to_string(Sj);
            std::cout << std::left
                      << std::setw(16) << ""
                      << std::setw(10) << 80
                      << std::setw(10) << tile_label.c_str()
                      << std::setw(8) << std::fixed << std::setprecision(1) << val_t
                      << std::setw(4) << unit_t
                      << std::setw(4) << " "
                      << std::setw(10) << cl_t
                      << std::setw(14) << std::fixed << std::setprecision(1) << best
                      << std::setw(10) << std::fixed << std::setprecision(2) << gops
                      << (ok ? "" : " FAIL")
                      << "\n";
        }

        std::cout << std::string(80, '-') << "\n";
    }
}

// ====================== 实验6：L1 分组建表查表 (BF16 子表) ======================

/** float → BF16 (取 float32 的高 16 位) */
static inline uint16_t float_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    return bits >> 16;
}

/** 预处理：为每行 A 构建组索引 */
struct GroupData {
    int G, step;
    int total_elements;
    std::vector<int> row_start;     // [N*G]
    std::vector<int> row_count;     // [N*G]
    std::vector<uint8_t> A_grouped;
    std::vector<int> k_indices;
};

static GroupData preprocess_groups(const uint8_t *A, int N, int L, int G) {
    GroupData d;
    d.G = G; d.step = 256 / G;
    d.row_start.assign(N * G, 0);
    d.row_count.assign(N * G, 0);

    for (int i = 0; i < N; ++i)
        for (int k = 0; k < L; ++k)
            d.row_count[i * G + A[i * L + k] / d.step]++;

    d.total_elements = 0;
    for (int i = 0; i < N; ++i)
        for (int g = 0; g < G; ++g) {
            d.row_start[i * G + g] = d.total_elements;
            d.total_elements += d.row_count[i * G + g];
        }

    d.A_grouped.resize(d.total_elements);
    d.k_indices.resize(d.total_elements);
    std::vector<int> cursor(N * G, 0);
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < L; ++k) {
            int g = A[i * L + k] / d.step;
            int pos = d.row_start[i * G + g] + cursor[i * G + g];
            d.A_grouped[pos] = A[i * L + k];
            d.k_indices[pos] = k;
            cursor[i * G + g]++;
        }
    return d;
}

/** 从完整 LUT 切出 G 个子表（BF16 格式，2B/entry） */
static void build_subtables_bf16(int G,
                                  std::vector<std::vector<uint16_t>> &subtables,
                                  const float *lut) {
    int step = 256 / G;
    subtables.resize(G);
    for (int g = 0; g < G; ++g) {
        subtables[g].resize(step * 256);
        for (int a = 0; a < step; ++a)
            for (int b = 0; b < 256; ++b)
                subtables[g][a * 256 + b] = float_to_bf16(lut[(g * step + a) * 256 + b]);
    }
}

/** G 核 SVE 分组建表查表（BF16 子表）：
    标量收集 B_T + BF16 子表查表，SVE 向量化累加 */
void lookup_sve_grouped_omp(const GroupData &data,
                            const std::vector<std::vector<uint16_t>> &subtables,
                            const uint8_t *B_T, float *C,
                            int N, int S, int L,
                            float *core_times, float &sync_time) {
    int G = data.G, step = data.step;

    #pragma omp parallel num_threads(G)
    {
        int g = omp_get_thread_num();
        const uint16_t *sub = subtables[g].data();
        int base = g * step;

        double t0 = omp_get_wtime();

        for (int i = 0; i < N; ++i) {
            int start = data.row_start[i * G + g];
            int count = data.row_count[i * G + g];
            const uint8_t *a_ptr = data.A_grouped.data() + start;
            const int *k_ptr = data.k_indices.data() + start;

            for (int j = 0; j < S; ++j) {
                const uint8_t *b_row = B_T + j * L;

                // 合并阶段：标量收集 B_T + BF16 子表查表 + 转 float
                // 子表全部 L1 命中，标量加载延迟低
                alignas(64) float local_vals[512];
                for (int t = 0; t < count; ++t) {
                    int b_val = b_row[k_ptr[t]];
                    int idx = (a_ptr[t] - base) * 256 + b_val;
                    uint32_t bits = (uint32_t)sub[idx] << 16;
                    memcpy(local_vals + t, &bits, 4);
                }

                // SVE 向量化累加
                svfloat32_t acc = svdup_n_f32(0.0f);
                int t = 0;
                svbool_t pg = svwhilelt_b32(t, count);
                while (svptest_any(svptrue_b32(), pg)) {
                    acc = svadd_f32_z(pg, acc, svld1_f32(pg, local_vals + t));
                    t += svcntw();
                    pg = svwhilelt_b32(t, count);
                }
                #pragma omp atomic
                C[i * S + j] += svaddv_f32(svptrue_b32(), acc);
            }
        }

        double t1 = omp_get_wtime();
        core_times[g] = (t1 - t0) * 1e6;

        #pragma omp barrier
        #pragma omp master
        sync_time = (omp_get_wtime() - t1) * 1e6;
    }
}

void exp6_l1_grouped_lut(const float *table, const uint8_t *A,
                          const uint8_t *B_T, const float *ref,
                          int N, int S, int L, int num_threads) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "实验6: L1 分组建表查表试验 (SVE, " << num_threads << " 核)\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "矩阵: " << N << "×" << L << " * " << S << "×" << L << "^T\n";
    std::cout << "L1d = " << L1D_KiB << " KiB, 子表格式: BF16 (2B)\n\n";

    auto gops = [&](double us) { return double(N) * S * L / us / 1e3; };

    // 基线：SVE + OpenMP 无分组
    omp_set_num_threads(num_threads);
    std::vector<float> C_bl(N * S);
    double base_us = bench([&]() {
        memset(C_bl.data(), 0, N * S * sizeof(float));
        lookup_sve_omp(table, A, B_T, C_bl.data(), N, S, L);
    });

    bool base_ok = verify(ref, C_bl.data(), N * S);

    std::cout << std::left
              << std::setw(10) << "方案"
              << std::setw(8) << "线程"
              << std::setw(14) << "子表"
              << std::setw(8) << "级别"
              << std::setw(18) << "计算(核min~max)"
              << std::setw(12) << "同步"
              << std::setw(14) << "总耗时(us)"
              << std::setw(10) << "GOP/s"
              << std::setw(10) << "正确"
              << "\n" << std::string(115, '-') << "\n";

    std::cout << std::left
              << std::setw(10) << "基线SVE"
              << std::setw(8) << num_threads
              << std::setw(14) << "256 KiB"
              << std::setw(8) << "L2"
              << std::setw(18) << "—"
              << std::setw(12) << "—"
              << std::setw(14) << std::fixed << std::setprecision(1) << base_us
              << std::setw(10) << std::fixed << std::setprecision(2) << gops(base_us)
              << std::setw(10) << (base_ok ? "OK" : "FAIL") << "\n";
    // 计算 BF16 参考值（全表 float->BF16->float，与分组建表精度一致）
    std::vector<float> lut_bf16(LUT_SIZE);
    for (int i = 0; i < LUT_SIZE; ++i) {
        uint32_t bits;
        memcpy(&bits, &table[i], 4);
        bits &= 0xFFFF0000u;
        float v;
        memcpy(&v, &bits, 4);
        lut_bf16[i] = v;
    }
    std::vector<float> C_bf16_ref(N * S);
    lookup_scalar(lut_bf16.data(), A, B_T, C_bf16_ref.data(), N, S, L);

    // G 值扫描：G=1 相当于单核分组查参考，G≥2 子表 ≤64K 尝试 L1 驻留
    int g_vals[] = {1, 2, 4, 8, 16, 32, 64};

    for (int Gi : g_vals) {
        if (Gi > omp_get_max_threads()) continue;
        if (Gi > N * S) continue;  // 至少每个线程一个元素

        GroupData gd = preprocess_groups(A, N, L, Gi);
        std::vector<std::vector<uint16_t>> subs;
        build_subtables_bf16(Gi, subs, table);

        int sub_kib = (256 / Gi) * 256 * 2 / 1024;
        const char *cl = cache_level(sub_kib, L1D_KiB, L2_KiB);

        std::vector<float> C_g(N * S, 0);
        std::vector<float> core_t(Gi);
        float sync_t = 0;

        // warmup 1 次
        lookup_sve_grouped_omp(gd, subs, B_T, C_g.data(), N, S, L,
                                core_t.data(), sync_t);

        // 正式测量（多次取最优总耗时）
        double best_total = 1e18;
        float best_sync = 0;
        std::vector<float> best_core(Gi);
        for (int run = 0; run < 3; ++run) {
            std::fill(C_g.begin(), C_g.end(), 0);
            std::fill(core_t.begin(), core_t.end(), 0);
            sync_t = 0;

            lookup_sve_grouped_omp(gd, subs, B_T, C_g.data(), N, S, L,
                                    core_t.data(), sync_t);

            double total = *std::max_element(core_t.begin(), core_t.end()) + sync_t;
            if (total < best_total) {
                best_total = total;
                best_sync = sync_t;
                best_core = core_t;
            }
        }

        float cmin = *std::min_element(best_core.begin(), best_core.end());
        float cmax = *std::max_element(best_core.begin(), best_core.end());
        double gops_v = gops(best_total);

        std::ostringstream ss_sub;
        if (sub_kib < 1024)
            ss_sub << sub_kib << " KiB";
        else
            ss_sub << std::fixed << std::setprecision(1) << (sub_kib / 1024.0) << " MiB";

        bool ok = verify(C_bf16_ref.data(), C_g.data(), N * S);

        std::ostringstream ss_comp;
        ss_comp << std::fixed << std::setprecision(1) << cmin << "~" << cmax;

        std::cout << std::left
                  << std::setw(10) << ("G=" + std::to_string(Gi)).c_str()
                  << std::setw(8) << Gi
                  << std::setw(14) << ss_sub.str()
                  << std::setw(8) << cl
                  << std::setw(18) << ss_comp.str()
                  << std::setw(12) << std::fixed << std::setprecision(1) << best_sync
                  << std::setw(14) << std::fixed << std::setprecision(1) << best_total
                  << std::setw(10) << std::fixed << std::setprecision(2) << gops_v
                  << std::setw(10) << (ok ? "OK" : "FAIL") << "\n";
    }
    std::cout << "\n";
}

// ====================== Main ======================

int main() {
    std::cout << "\n" << std::string(70, '#') << "\n";
    std::cout << "  FP8 查表法矩阵乘法 — 服务器性能测试\n";
    std::cout << "  平台: HiSilicon Kunpeng, SVE\n";
    std::cout << "  SVE vector width: svcntw()=" << svcntw() << ", svcntb()=" << svcntb() << "\n";
    std::cout << "  Cache: L1d=" << L1D_KiB << "KiB  L2=" << L2_KiB << "KiB  L3=70MiB/Node\n";
    std::cout << "  LUT: 256×256 float = 256 KiB (> L1d, 永远在 L2+)\n";
    print_numa_info();
    std::cout << std::string(70, '#') << "\n";

    srand(42);

    // ========= 小矩阵实验 (与 x86 平台对比，验证 SVE / Tiling / L1 分组) =========
    const int N1 = 128, S1 = 328, L1 = 512;

    std::vector<uint8_t> A(N1 * L1), B_T(S1 * L1);
    std::vector<float> LUT(LUT_SIZE);
    std::vector<float> C_ref(N1 * S1);

    gen_lut(LUT.data());
    fill_random(A.data(), N1 * L1);
    fill_random(B_T.data(), S1 * L1);

    // 参考结果
    lookup_scalar(LUT.data(), A.data(), B_T.data(), C_ref.data(), N1, S1, L1);
    std::cout << "\n参考结果 C[0][0] = " << C_ref[0] << "\n";

    // 实验1: SVE 加速比 (1 核)
    exp1_sve_speedup(LUT.data(), A.data(), B_T.data(), N1, S1, L1);

    // 实验2: Tile 扫描 (SVE, 80 核 = 1 NUMA node)
    exp2_tile_sweep(LUT.data(), A.data(), B_T.data(), C_ref.data(), N1, S1, L1, 80);

    // 实验3: NUMA node 内多核扩展 (SVE, 使用首节点 CPU)
    {
        int numa_node = 0;
        auto cpus = cpus_on_node(numa_node);
        std::cout << "  Node " << numa_node << " CPUs: " << cpus.size() << " 个 ("
                  << cpus.front() << "~" << cpus.back() << ")\n";
        exp3_numa_scaling(LUT.data(), A.data(), B_T.data(), C_ref.data(),
                         N1, S1, L1, numa_node, cpus);
    }

    // 实验6: L1 分组建表查表试验 (80 核, SVE)
    exp6_l1_grouped_lut(LUT.data(), A.data(), B_T.data(), C_ref.data(),
                        N1, S1, L1, 80);

    // ========= 大矩阵实验 (跨 NUMA + Cache 边界) =========

    // 实验4: 跨 NUMA 扩展 (用固定矩阵测试 1~320 核)
    exp4_cross_numa(LUT.data(), 512, 2048, 512);

    // 实验5: 矩阵规模扩展 (80 核, 对比 tile 效果)
    exp5_matrix_scaling(LUT.data());

    std::cout << "\n测试完成!\n";
    return 0;
}
