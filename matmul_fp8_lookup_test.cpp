#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <cstdint>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <omp.h>
#include <sstream>
#include <algorithm>

using TimePoint = std::chrono::high_resolution_clock::time_point;
using namespace std::chrono;

// ======================== 数据加载/生成 ========================

template <typename T>
void load_or_generate(const std::string &path, T *data, size_t size) {
    std::ifstream is(path, std::ios::binary);
    if (is.is_open()) {
        is.read(reinterpret_cast<char*>(data), size * sizeof(T));
        size_t read_count = is.gcount() / sizeof(T);
        for (size_t i = read_count; i < size; i++)
            data[i] = static_cast<T>(rand() % 256);
        if (read_count == size) return;
    }
    for (size_t i = 0; i < size; i++)
        data[i] = static_cast<T>(rand() % 256);
}

void gen_lut(float *lut) {
    for (int i = 0; i < 256; i++)
        for (int j = 0; j < 256; j++)
            lut[i * 256 + j] = sinf(i * 0.1f) * cosf(j * 0.1f);
}

// ======================== 计算核心 ========================

/** 标量版 FP8 查表——无分块、无 OpenMP（串行基线） */
void lookup_scalar(const float *table, const uint8_t *A, const uint8_t *B_T,
                   float *C, int N, int S, int L) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < S; ++j) {
            float sum = 0.0f;
            const uint8_t *rA = A + i * L;
            const uint8_t *rB = B_T + j * L;
            for (int k = 0; k < L; ++k) {
                sum += table[(static_cast<int>(rA[k]) << 8) | rB[k]];
            }
            C[i * S + j] = sum;
        }
    }
}

/** 标量版 FP8 查表——OpenMP 行并行（无分块） */
void lookup_scalar_omp(const float *table, const uint8_t *A, const uint8_t *B_T,
                       float *C, int N, int S, int L) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < S; ++j) {
            float sum = 0.0f;
            const uint8_t *rA = A + i * L;
            const uint8_t *rB = B_T + j * L;
            for (int k = 0; k < L; ++k) {
                sum += table[(static_cast<int>(rA[k]) << 8) | rB[k]];
            }
            C[i * S + j] = sum;
        }
    }
}

/** 标量版 FP8 查表——分块 + OpenMP */
void lookup_scalar_tiled_omp(const float *table, const uint8_t *A, const uint8_t *B_T,
                              float *C, int N, int S, int L, int Ni, int Sj) {
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ii = 0; ii < N; ii += Ni) {
        for (int jj = 0; jj < S; jj += Sj) {
            int i_end = std::min(ii + Ni, N);
            int j_end = std::min(jj + Sj, S);
            for (int i = ii; i < i_end; ++i) {
                for (int j = jj; j < j_end; ++j) {
                    float sum = 0.0f;
                    const uint8_t *rA = A + i * L;
                    const uint8_t *rB = B_T + j * L;
                    for (int k = 0; k < L; ++k) {
                        sum += table[(static_cast<int>(rA[k]) << 8) | rB[k]];
                    }
                    C[i * S + j] = sum;
                }
            }
        }
    }
}

// ======================== Cache 分析工具 ========================

/** 估算单个 tile 的工作集大小（KiB） */
int calc_tile_workset_kib(int Ni, int Sj, int L, int lut_kib = 256) {
    int a_part   = Ni * L * 1;         // A tile
    int b_part   = Sj * L * 1;         // B_T tile
    int c_part   = Ni * Sj * 4;        // C tile
    return (a_part + b_part + c_part + lut_kib * 1024) / 1024;
}

const char* cache_level(int workset_kib, int L1d_kib, int L2_kib) {
    if (workset_kib <= 0) return "N/A";
    if (workset_kib <= L1d_kib) return "L1";
    if (workset_kib <= L2_kib)  return "L2";
    return "L3/DRAM";
}

double format_size_kib(int kib) {
    if (kib < 1024) return kib;
    return kib / 1024.0;
}

const char* size_unit(int kib) {
    if (kib < 1024) return "KiB";
    return "MiB";
}

// ======================== 实验函数 ========================

const int L1D_KiB = 48;
const int L2_KiB  = 1280;

void experiment_thread_scaling(const float *table, const uint8_t *A, const uint8_t *B_T,
                                float *C, float *ref, int N, int S, int L) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "实验1: 多核扩展性（全量矩阵，无分块）\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << "矩阵: " << N << "x" << L << " * " << S << "x" << L << "^T\n";
    std::cout << "单线程工作集: ~" << calc_tile_workset_kib(N, S, L) << " KiB\n\n";

    std::cout << std::left
              << std::setw(10) << "Threads"
              << std::setw(15) << "Time(us)"
              << std::setw(13) << "Speedup"
              << std::setw(12) << "Efficiency"
              << std::setw(12) << "GOP/s"
              << std::setw(10) << "Valid?"
              << "\n" << std::string(72, '-') << "\n";

    double single_time = 0;
    int max_threads = omp_get_max_threads();

    std::vector<int> thread_counts = {1, 2, 4, 6, 8, 10};
    if (max_threads > 10) thread_counts.push_back(max_threads);

    for (int nt : thread_counts) {
        omp_set_num_threads(nt);

        auto start = high_resolution_clock::now();
        lookup_scalar_omp(table, A, B_T, C, N, S, L);
        auto end = high_resolution_clock::now();
        double us = duration_cast<nanoseconds>(end - start).count() / 1000.0;

        if (nt == 1) single_time = us;

        bool ok = true;
        for (int i = 0; i < N * S; ++i) {
            if (fabsf(ref[i] - C[i]) > 1e-3f) { ok = false; break; }
        }

        double speedup = single_time / us;
        double eff = speedup / nt * 100;

        std::ostringstream ss_speedup, ss_eff, ss_gops;
        ss_speedup << std::fixed << std::setprecision(2) << speedup << "x";
        ss_eff     << std::fixed << std::setprecision(0) << eff << "%";
        ss_gops    << std::fixed << std::setprecision(2) << (N * S * 512.0 / us / 1e3);
        std::cout << std::left
                  << std::setw(10) << nt
                  << std::setw(15) << std::fixed << std::setprecision(1) << us
                  << std::setw(13) << ss_speedup.str()
                  << std::setw(12) << ss_eff.str()
                  << std::setw(12) << ss_gops.str()
                  << std::setw(10) << (ok ? "OK" : "FAIL")
                  << "\n";
    }
}

void experiment_tile_sweep(const float *table, const uint8_t *A, const uint8_t *B_T,
                            float *C, float *ref, int N, int S, int L, int num_threads) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "实验2: Tile 尺寸扫描（" << num_threads << " 线程）\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << "矩阵: " << N << "x" << L << " * " << S << "x" << L << "^T\n";
    std::cout << "L1d = " << L1D_KiB << " KiB, L2/核 = " << L2_KiB << " KiB\n\n";

    // 本实验数据能全部放进 L2，主要观察:
    // 1. LUT(256K) > L1d → 永远无法 L1 命中
    // 2. 不同 tile 大小对 LUT 在 L2 中驻留率的影响
    std::cout << std::left
              << std::setw(8) << "Ni"
              << std::setw(8) << "Sj"
              << std::setw(16) << "TileWorkset"
              << std::setw(12) << "CacheLvl"
              << std::setw(14) << "Time(us)"
              << std::setw(12) << "GOP/s"
              << std::setw(12) << "Valid?"
              << "\n" << std::string(70, '-') << "\n";

    omp_set_num_threads(num_threads);
    double ref_time = 0;

    // Tile 组合: 行分块 × 列分块
    int ni_vals[] = {2, 4, 8, 16, 32, 64, 128};
    int sj_vals[] = {8, 16, 32, 64, 128, 256, 328};

    for (int ni : ni_vals) {
        for (int sj : sj_vals) {
            if (ni > N || sj > S) continue;

            int workset = calc_tile_workset_kib(ni, sj, L);
            const char *cl = cache_level(workset, L1D_KiB, L2_KiB);

            float ws_kib = format_size_kib(workset);
            const char *unit = size_unit(workset);

            // 多次运行取最优
            double best = 1e18;
            int iters = 3;
            for (int t = 0; t < iters; ++t) {
                memset(C, 0, N * S * sizeof(float));
                auto start = high_resolution_clock::now();
                lookup_scalar_tiled_omp(table, A, B_T, C, N, S, L, ni, sj);
                auto end = high_resolution_clock::now();
                double us = duration_cast<nanoseconds>(end - start).count() / 1000.0;
                if (us < best) best = us;
            }

            if (ni == N && sj == S) ref_time = best;

            // 验证
            bool ok = true;
            for (int i = 0; i < N * S; ++i) {
                if (fabsf(ref[i] - C[i]) > 1e-3f) { ok = false; break; }
            }

            double gops = N * S * 512.0 / best / 1e3;
            std::cout << std::left
                      << std::setw(8) << ni
                      << std::setw(8) << sj
                      << std::setw(12) << std::fixed << std::setprecision(1) << ws_kib
                      << std::setw(4) << unit
                      << std::setw(12) << cl
                      << std::setw(14) << std::fixed << std::setprecision(1) << best
                      << std::setw(12) << std::fixed << std::setprecision(2) << gops
                      << std::setw(12) << (ok ? "OK" : "FAIL")
                      << "\n";
        }
    }
}

void experiment_matrix_scaling(const float *table, float *temp_result) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "实验3: 矩阵规模扩展 × Tile 优化效果\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << "L1d = " << L1D_KiB << " KiB, L2/核 = " << L2_KiB << " KiB, L3共享 = 25 MiB\n\n";

    struct ScaleInfo {
        int N, S, L;
        const char* label;
    };
    ScaleInfo scales[] = {
        {128,  328,  512, "1x  (128×328)"},
        {256,  656,  512, "2x  (256×656)"},
        {512,  1312, 512, "4x  (512×1312)"},
        {1024, 2624, 512, "8x  (1024×2624)"},
    };

    std::cout << std::left
              << std::setw(16) << "Scale"
              << std::setw(12) << "Threads"
              << std::setw(8) << "Tiling"
              << std::setw(18) << "WorkSet/Core"
              << std::setw(12) << "Cache"
              << std::setw(14) << "Time(us)"
              << std::setw(12) << "GOP/s"
              << "\n" << std::string(80, '-') << "\n";

    for (auto &s : scales) {
        int N = s.N, S = s.S, L = s.L;

        // 生成数据
        std::vector<uint8_t> A(N * L), B_T(S * L);
        std::vector<float> C_full(N * S);
        for (int i = 0; i < N * L; ++i) A[i] = rand() % 256;
        for (int i = 0; i < S * L; ++i) B_T[i] = rand() % 256;

        // 分配每个 scale 的独立缓冲区
        std::vector<float> ref_full(N * S);
        std::vector<float> result_full(N * S);

        // 参考结果（1线程）
        omp_set_num_threads(1);
        lookup_scalar(table, A.data(), B_T.data(), ref_full.data(), N, S, L);

        // --- 1 线程，无分块 ---
        {
            int workset = calc_tile_workset_kib(N, S, L);
            float ws_kib = format_size_kib(workset);
            const char *unit = size_unit(workset);
            const char *cl = cache_level(workset, L1D_KiB, L2_KiB);

            omp_set_num_threads(1);
            auto start = high_resolution_clock::now();
            lookup_scalar(table, A.data(), B_T.data(), C_full.data(), N, S, L);
            auto end = high_resolution_clock::now();
            double us = duration_cast<nanoseconds>(end - start).count() / 1000.0;

            // 验证
            bool ok = true;
            for (int i = 0; i < N * S; ++i)
                if (fabsf(ref_full[i] - C_full[i]) > 1e-3f) { ok = false; break; }

            double gops = N * S * 512.0 / us / 1e3;
            std::cout << std::left
                      << std::setw(16) << s.label
                      << std::setw(12) << 1
                      << std::setw(8) << "none"
                      << std::setw(14) << std::fixed << std::setprecision(1) << ws_kib
                      << std::setw(4) << unit
                      << std::setw(12) << cl
                      << std::setw(14) << std::fixed << std::setprecision(1) << us
                      << std::setw(12) << std::fixed << std::setprecision(2) << gops
                      << "\n";
        }

        // --- 10 线程，无分块 ---
        {
            int workset = calc_tile_workset_kib(N, S, L);
            float ws_kib = format_size_kib(workset);
            const char *unit = size_unit(workset);
            // 10 核共享 L3 = 25 MiB → 每核有效 ~2.5 MiB
            const char *cl = (workset <= 2500) ? "L3" : "DRAM";

            omp_set_num_threads(10);
            auto start = high_resolution_clock::now();
            lookup_scalar_omp(table, A.data(), B_T.data(), C_full.data(), N, S, L);
            auto end = high_resolution_clock::now();
            double us = duration_cast<nanoseconds>(end - start).count() / 1000.0;

            bool ok = true;
            for (int i = 0; i < N * S; ++i)
                if (fabsf(ref_full[i] - C_full[i]) > 1e-3f) { ok = false; break; }

            double gops = N * S * 512.0 / us / 1e3;
            std::cout << std::left
                      << std::setw(16) << ""
                      << std::setw(12) << 10
                      << std::setw(8) << "none"
                      << std::setw(14) << std::fixed << std::setprecision(1) << ws_kib
                      << std::setw(4) << unit
                      << std::setw(12) << cl
                      << std::setw(14) << std::fixed << std::setprecision(1) << us
                      << std::setw(12) << std::fixed << std::setprecision(2) << gops
                      << "\n";
        }

        // --- 10 线程 + Tile 优化 ---
        {
            // 选择一个合适的 tile (Ni=32, Sj=128) → ~432 KiB → L2
            int Ni = std::min(32, N);
            int Sj = std::min(128, S);
            int workset = calc_tile_workset_kib(Ni, Sj, L);
            const char *cl = cache_level(workset, L1D_KiB, L2_KiB);
            float ws_kib = format_size_kib(workset);
            const char *unit = size_unit(workset);

            omp_set_num_threads(10);
            auto start = high_resolution_clock::now();
            lookup_scalar_tiled_omp(table, A.data(), B_T.data(), C_full.data(), N, S, L, Ni, Sj);
            auto end = high_resolution_clock::now();
            double us = duration_cast<nanoseconds>(end - start).count() / 1000.0;

            bool ok = true;
            for (int i = 0; i < N * S; ++i)
                if (fabsf(ref_full[i] - C_full[i]) > 1e-3f) { ok = false; break; }

            double gops = N * S * 512.0 / us / 1e3;
            std::cout << std::left
                      << std::setw(16) << ""
                      << std::setw(12) << 10
                      << std::setw(8) << (std::to_string(Ni)+"x"+std::to_string(Sj)).c_str()
                      << std::setw(14) << std::fixed << std::setprecision(1) << ws_kib
                      << std::setw(4) << unit
                      << std::setw(12) << cl
                      << std::setw(14) << std::fixed << std::setprecision(1) << us
                      << std::setw(12) << std::fixed << std::setprecision(2) << gops
                      << "\n";
        }

        std::cout << std::string(80, '-') << "\n";
    }
}

// ======================== 实验4: L1 分组建表 ========================

/** 预处理：为每行 A 构建组索引 */
struct GroupData {
    int G;
    int step;                           // 256 / G
    int total_elements;                 // 总元素数（= N×L）
    std::vector<int> row_start;         // [N*G] per-row per-group 起始偏移
    std::vector<int> row_count;         // [N*G] per-row per-group 元素数
    std::vector<uint8_t> A_grouped;     // 按组重排的 A 值
    std::vector<int> k_indices;         // 原始 k 索引
};

GroupData preprocess_groups(const uint8_t *A, int N, int L, int G) {
    GroupData d;
    d.G = G;
    d.step = 256 / G;
    d.row_start.resize(N * G, 0);
    d.row_count.resize(N * G, 0);

    // 计数
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < L; ++k)
            d.row_count[i * G + A[i * L + k] / d.step]++;

    // 计算起始偏移
    d.total_elements = 0;
    for (int i = 0; i < N; ++i)
        for (int g = 0; g < G; ++g) {
            d.row_start[i * G + g] = d.total_elements;
            d.total_elements += d.row_count[i * G + g];
        }

    // 填充
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

/** 从完整 LUT 切出 G 个子表 */
void build_subtables(int G, std::vector<std::vector<float>> &subtables,
                     const float *lut) {
    int step = 256 / G;
    subtables.resize(G);
    for (int g = 0; g < G; ++g) {
        subtables[g].resize(step * 256);
        for (int a = 0; a < step; ++a)
            for (int b = 0; b < 256; ++b)
                subtables[g][a * 256 + b] = lut[(g * step + a) * 256 + b];
    }
}

/** G 核协作查表：每核持一个子表，共同计算同一个 (i,j) */
void lookup_scalar_grouped_omp(const GroupData &data,
                               const std::vector<std::vector<float>> &subtables,
                               const uint8_t *B_T, float *C,
                               int N, int S, int L,
                               float *core_times, float &sync_time) {
    int G = data.G;
    int step = data.step;

    #pragma omp parallel num_threads(G)
    {
        int g = omp_get_thread_num();
        const float *sub = subtables[g].data();
        int base = g * step;  // 此组的 idxA 起始值

        double t0 = omp_get_wtime();

        for (int i = 0; i < N; ++i) {
            int start = data.row_start[i * G + g];
            int count = data.row_count[i * G + g];
            const uint8_t *a_ptr = data.A_grouped.data() + start;
            const int *k_ptr = data.k_indices.data() + start;

            for (int j = 0; j < S; ++j) {
                float sum = 0.0f;
                const uint8_t *b_row = B_T + j * L;
                for (int t = 0; t < count; ++t) {
                    sum += sub[(a_ptr[t] - base) * 256 + b_row[k_ptr[t]]];
                }
                #pragma omp atomic
                C[i * S + j] += sum;
            }
        }

        double t1 = omp_get_wtime();
        core_times[g] = (t1 - t0) * 1e6;

        #pragma omp barrier
        #pragma omp master
        sync_time = (omp_get_wtime() - t1) * 1e6;
    }
}

void experiment_l1_grouped_lut(const float *table, const uint8_t *A,
                                const uint8_t *B_T, const float *ref,
                                int N, int S, int L) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "实验4: L1 分组建表查表试验 (多核协作)\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "矩阵: " << N << "×" << L << " * " << S << "×" << L << "^T\n";
    std::cout << "L1d = " << L1D_KiB << " KiB, 子表格式: float (4B)\n\n";

    // 基线：标准 OMP 版本
    int max_th = std::min(10, omp_get_max_threads());
    omp_set_num_threads(max_th);
    std::vector<float> C_bl(N * S);
    auto start = high_resolution_clock::now();
    lookup_scalar_omp(table, A, B_T, C_bl.data(), N, S, L);
    auto end = high_resolution_clock::now();
    double base_us = duration_cast<nanoseconds>(end - start).count() / 1000.0;
    bool base_ok = true;
    for (int i = 0; i < N * S; ++i)
        if (fabsf(ref[i] - C_bl[i]) > 1e-3f) { base_ok = false; break; }

    // GOP/s = N*S*L / time_us / 1e3  (与实验1/2/3保持一致)
    auto gops = [&](double us) { return double(N) * S * L / us / 1e3; };

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
              << "\n" << std::string(110, '-') << "\n";

    std::cout << std::left
              << std::setw(10) << "基线OMP"
              << std::setw(8) << max_th
              << std::setw(14) << "256 KiB"
              << std::setw(8) << "L3"
              << std::setw(18) << "—"
              << std::setw(12) << "—"
              << std::setw(14) << std::fixed << std::setprecision(1) << base_us
              << std::setw(10) << std::fixed << std::setprecision(2) << gops(base_us)
              << std::setw(10) << (base_ok ? "OK" : "FAIL") << "\n";

    // 测试不同 G
    int g_vals[] = {1, 2, 4, 8};

    for (int Gi : g_vals) {
        if (Gi > omp_get_max_threads()) continue;

        // 预处理
        GroupData gd = preprocess_groups(A, N, L, Gi);
        std::vector<std::vector<float>> subs;
        build_subtables(Gi, subs, table);

        int sub_kib = (256 / Gi) * 256 * 4 / 1024;
        const char *cl = cache_level(sub_kib, L1D_KiB, L2_KiB);

        // 运行
        std::vector<float> C_g(N * S, 0);
        std::vector<float> core_t(Gi);
        float sync_t = 0;

        // warmup
        lookup_scalar_grouped_omp(gd, subs, B_T, C_g.data(), N, S, L,
                                   core_t.data(), sync_t);

        // 正式测量（多次取最优总耗时）
        double best_total = 1e18;
        float best_sync = 0;
        std::vector<float> best_core(Gi);
        for (int run = 0; run < 3; ++run) {
            std::fill(C_g.begin(), C_g.end(), 0);
            std::fill(core_t.begin(), core_t.end(), 0);
            sync_t = 0;

            lookup_scalar_grouped_omp(gd, subs, B_T, C_g.data(), N, S, L,
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
        std::ostringstream ss_sub;
        if (sub_kib < 1024)
            ss_sub << sub_kib << " KiB";
        else
            ss_sub << std::fixed << std::setprecision(1) << (sub_kib / 1024.0) << " MiB";

        double gops_v = gops(best_total);

        bool ok = true;
        for (int i = 0; i < N * S; ++i)
            if (fabsf(ref[i] - C_g[i]) > 1e-3f) { ok = false; break; }

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

int main() {
    const int N = 128, S = 328, L = 512;

    std::cout << "\n" << std::string(60, '#') << "\n";
    std::cout << "  FP8 查表法矩阵乘法 Cache 性能测试\n";
    std::cout << "  CPU: Intel i7-12700 (10核) @ Hyper-V\n";
    std::cout << "  L1d = " << L1D_KiB << " KiB/核, L2 = " << L2_KiB << " KiB/核, L3 = 25 MiB\n";
    std::cout << "  查找表: 256×256 float = 256 KiB (> L1d, 永远在 L2+)\n";
    std::cout << std::string(60, '#') << "\n";

    // 分配内存
    std::vector<uint8_t> A(N * L), B_T(S * L);
    std::vector<float> LUT(256 * 256);
    std::vector<float> C_ref(N * S), C_result(N * S);

    // 生成数据
    srand(42);
    load_or_generate("./input/matrix_a_8.bin", A.data(), N * L);
    load_or_generate("./input/matrix_b_8.bin", B_T.data(), S * L);
    load_or_generate("./input/look_up_table_fp32.bin", LUT.data(), 256 * 256);
    gen_lut(LUT.data()); // override with synthetic if needed

    // 计算参考结果（串行）
    lookup_scalar(LUT.data(), A.data(), B_T.data(), C_ref.data(), N, S, L);
    std::cout << "\n参考结果 (C[0][0]) = " << C_ref[0] << std::endl;

    // ===== 实验1: 多核扩展性 =====
    experiment_thread_scaling(LUT.data(), A.data(), B_T.data(), C_result.data(), C_ref.data(), N, S, L);

    // ===== 实验2: Tile 尺寸扫描 =====
    experiment_tile_sweep(LUT.data(), A.data(), B_T.data(), C_result.data(), C_ref.data(), N, S, L, 10);

    // ===== 实验3: 矩阵规模扩展 =====
    experiment_matrix_scaling(LUT.data(), C_result.data());

    // ===== 实验4: L1 分组建表查表试验 =====
    experiment_l1_grouped_lut(LUT.data(), A.data(), B_T.data(), C_ref.data(), N, S, L);

    std::cout << "\n测试完成!\n";
    return 0;
}
