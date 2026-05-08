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
#include <sys/ioctl.h>
#include <numa.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>

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

const char* cache_level(int workset_kib, int L1d, int L2) {
    if (workset_kib <= L1d) return "L1";
    if (workset_kib <= L2)  return "L2";
    return "L3+";
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

/** Linux perf_event_open 封装，用于采集 ARM PMU 硬件事件 */
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
        pea.disabled = 1;      // start disabled
        pea.pinned = 1;         // require counter to be on PMU
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
        if (fd_ >= 0) {
            if (::read(fd_, &val, sizeof(val)) != sizeof(val)) return 0;
        }
        return val;
    }
};

/** ARMv8.0 PMU 事件编码（鲲鹏 920 兼容） */
namespace ArmPmu {
    // Standard ARMv8 events
    constexpr uint64_t L1D_CACHE        = 0x04;  // L1 data cache access
    constexpr uint64_t L1D_CACHE_REFILL = 0x03;  // L1 data cache refill (miss)
    // Implementation-defined (Kunpeng should support)
    constexpr uint64_t L2D_CACHE        = 0x16;  // L2 data cache access
    constexpr uint64_t L2D_CACHE_REFILL = 0x17;  // L2 data cache refill
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
                            float *core_times, float &sync_time,
                            float *scalar_times = nullptr,
                            float *sve_times = nullptr) {
    int G = data.G, step = data.step;

    #pragma omp parallel num_threads(G)
    {
        int g = omp_get_thread_num();
        const uint16_t *sub = subtables[g].data();
        int base = g * step;

        double t_scalar = 0, t_sve = 0;
        double t0 = omp_get_wtime();

        for (int i = 0; i < N; ++i) {
            int start = data.row_start[i * G + g];
            int count = data.row_count[i * G + g];
            const uint8_t *a_ptr = data.A_grouped.data() + start;
            const int *k_ptr = data.k_indices.data() + start;

            for (int j = 0; j < S; ++j) {
                const uint8_t *b_row = B_T + j * L;

                double ts0 = omp_get_wtime();
                // 阶段1：标量收集 B_T + BF16 子表查表 + 转 float
                alignas(64) float local_vals[512];
                for (int t = 0; t < count; ++t) {
                    int b_val = b_row[k_ptr[t]];
                    int idx = (a_ptr[t] - base) * 256 + b_val;
                    uint32_t bits = (uint32_t)sub[idx] << 16;
                    memcpy(local_vals + t, &bits, 4);
                }
                t_scalar += omp_get_wtime() - ts0;

                double ts1 = omp_get_wtime();
                // 阶段2：SVE 向量化累加 + atomic
                svfloat32_t acc = svdup_n_f32(0.0f);
                int t = 0;
                svbool_t pg = svwhilelt_b32(t, count);
                while (svptest_any(svptrue_b32(), pg)) {
                    acc = svadd_f32_m(pg, acc, svld1_f32(pg, local_vals + t));
                    t += svcntw();
                    pg = svwhilelt_b32(t, count);
                }
                #pragma omp atomic
                C[i * S + j] += svaddv_f32(svptrue_b32(), acc);
                t_sve += omp_get_wtime() - ts1;
            }
        }

        double t1 = omp_get_wtime();
        core_times[g] = (t1 - t0) * 1e6;
        if (scalar_times) scalar_times[g] = t_scalar * 1e6;
        if (sve_times)    sve_times[g]    = t_sve * 1e6;

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
    std::cout << "L1d = " << L1D_KiB << " KiB, 子表格式: BF16 (2B)\n";
    std::cout << "Warmup=" << 100 << ", 采样=" << 1000 << "\n\n";

    auto gops = [&](double us) { return double(N) * S * L / us / 1e3; };

    // ——— 基线：SVE + OpenMP 无分组 ———
    omp_set_num_threads(num_threads);
    std::vector<float> C_bl(N * S);
    double base_us = bench([&]() {
        memset(C_bl.data(), 0, N * S * sizeof(float));
        lookup_sve_omp(table, A, B_T, C_bl.data(), N, S, L);
    });
    bool base_ok = verify(ref, C_bl.data(), N * S);

    // ——— BF16 参考值 ———
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

    // ——— 输出表头 ———
    std::cout << std::left
              << std::setw(10) << "方案"
              << std::setw(8) << "线程"
              << std::setw(14) << "子表"
              << std::setw(8) << "级别"
              << std::setw(18) << "计算(核min~max)"
              << std::setw(12) << "同步"
              << std::setw(14) << "标量相位"
              << std::setw(14) << "SVE相位"
              << std::setw(14) << "预处理"
              << std::setw(12) << "L1-miss%"
              << std::setw(12) << "L2-miss%"
              << std::setw(18) << "总耗时mean±sd"
              << std::setw(12) << "总耗时min"
              << std::setw(10) << "GOP/s"
              << std::setw(10) << "正确"
              << "\n" << std::string(175, '-') << "\n";

    // 基线行
    std::cout << std::left
              << std::setw(10) << "基线SVE"
              << std::setw(8) << num_threads
              << std::setw(14) << "256 KiB"
              << std::setw(8) << "L2"
              << std::setw(18) << "—"
              << std::setw(12) << "—"
              << std::setw(14) << "—"
              << std::setw(14) << "—"
              << std::setw(14) << "—"
              << std::setw(12) << "—"
              << std::setw(12) << "—"
              << std::setw(18) << std::fixed << std::setprecision(1) << base_us
              << std::setw(12) << "—"
              << std::setw(10) << std::fixed << std::setprecision(2) << gops(base_us)
              << std::setw(10) << (base_ok ? "OK" : "FAIL") << "\n";

    // ——— 扫描 G ———
    constexpr int G_VALS[] = {1, 2, 4, 8, 16, 32, 64};
    constexpr int WARMUP = 100;
    constexpr int ITERS = 10000;

    for (int Gi : G_VALS) {
        if (Gi > omp_get_max_threads()) continue;
        if (Gi > N * S) continue;

        // ── 预处理（计时） ──
        double t_prep = omp_get_wtime();
        GroupData gd = preprocess_groups(A, N, L, Gi);
        std::vector<std::vector<uint16_t>> subs;
        build_subtables_bf16(Gi, subs, table);
        double prep_us = (omp_get_wtime() - t_prep) * 1e6;

        int sub_kib = (256 / Gi) * 256 * 2 / 1024;
        const char *cl = cache_level(sub_kib, L1D_KiB, L2_KiB);

        // ── Perf 计数器（打开失败则降级） ──
        PerfCounter pc_l1_acc(ArmPmu::L1D_CACHE);
        PerfCounter pc_l1_miss(ArmPmu::L1D_CACHE_REFILL);
        PerfCounter pc_l2_acc(ArmPmu::L2D_CACHE);
        PerfCounter pc_l2_miss(ArmPmu::L2D_CACHE_REFILL);
        bool perf_ok = pc_l1_acc.ok() && pc_l1_miss.ok();

        // ── 复用缓冲区 ──
        std::vector<float> C_g(N * S, 0);
        std::vector<float> core_t(Gi);
        std::vector<float> scalar_t(Gi), sve_t(Gi);
        float sync_t = 0;

        // ── Warmup ──
        for (int w = 0; w < WARMUP; ++w) {
            std::fill(C_g.begin(), C_g.end(), 0);
            lookup_sve_grouped_omp(gd, subs, B_T, C_g.data(), N, S, L,
                                   core_t.data(), sync_t);
        }

        // ── 测量 ──
        // 在线统计：总耗时
        double sum_total = 0, sum_total2 = 0;
        double min_total = 1e18, max_total = 0;
        // 同步时间
        double sum_sync = 0;
        // 阶段耗时（累计后取平均）
        double sum_scalar = 0, sum_sve = 0;
        // 最佳单次（用于输出 compute min~max）
        double best_total = 1e18;
        float best_sync = 0;
        std::vector<float> best_core(Gi);

        // 启动 perf 计数器（从第一次测量开始累计）
        if (perf_ok) {
            pc_l1_acc.reset(); pc_l1_acc.enable();
            pc_l1_miss.reset(); pc_l1_miss.enable();
            if (pc_l2_acc.ok()) { pc_l2_acc.reset(); pc_l2_acc.enable(); }
            if (pc_l2_miss.ok()) { pc_l2_miss.reset(); pc_l2_miss.enable(); }
        }

        for (int iter = 0; iter < ITERS; ++iter) {
            std::fill(C_g.begin(), C_g.end(), 0);
            std::fill(core_t.begin(), core_t.end(), 0);
            std::fill(scalar_t.begin(), scalar_t.end(), 0);
            std::fill(sve_t.begin(), sve_t.end(), 0);
            sync_t = 0;

            lookup_sve_grouped_omp(gd, subs, B_T, C_g.data(), N, S, L,
                                   core_t.data(), sync_t,
                                   scalar_t.data(), sve_t.data());

            // 累加阶段耗时（取每核 max，因为最慢核决定总时间）
            double max_scalar = *std::max_element(scalar_t.begin(), scalar_t.end());
            double max_sve = *std::max_element(sve_t.begin(), sve_t.end());
            sum_scalar += max_scalar;
            sum_sve += max_sve;

            double total = *std::max_element(core_t.begin(), core_t.end()) + sync_t;

            // 更新在线统计
            sum_total += total;
            sum_total2 += total * total;
            if (total < min_total) min_total = total;
            if (total > max_total) max_total = total;
            sum_sync += sync_t;

            // 记录最优单次（用于输出计算核时间）
            if (total < best_total) {
                best_total = total;
                best_sync = sync_t;
                best_core = core_t;
            }
        }

        // 停止 perf 计数器
        if (perf_ok) {
            pc_l1_acc.disable();
            pc_l1_miss.disable();
            if (pc_l2_acc.ok()) pc_l2_acc.disable();
            if (pc_l2_miss.ok()) pc_l2_miss.disable();
        }

        // ── 统计计算 ──
        double mean_total = sum_total / ITERS;
        double var_total = (sum_total2 - sum_total * mean_total) / (ITERS - 1);
        double sd_total = std::sqrt(std::max(0.0, var_total));
        double avg_sync = sum_sync / ITERS;
        double avg_scalar = sum_scalar / ITERS;
        double avg_sve = sum_sve / ITERS;

        float cmin = *std::min_element(best_core.begin(), best_core.end());
        float cmax = *std::max_element(best_core.begin(), best_core.end());
        double gops_min = gops(min_total);

        // ── Cache miss rate ──
        double l1_miss_rate = -1, l2_miss_rate = -1;
        if (perf_ok) {
            uint64_t l1_a = pc_l1_acc.read();
            uint64_t l1_m = pc_l1_miss.read();
            if (l1_a > 0) l1_miss_rate = 100.0 * l1_m / l1_a;

            if (pc_l2_acc.ok() && pc_l2_miss.ok()) {
                uint64_t l2_a = pc_l2_acc.read();
                uint64_t l2_m = pc_l2_miss.read();
                if (l2_a > 0) l2_miss_rate = 100.0 * l2_m / l2_a;
            }
        }

        bool ok = verify(C_bf16_ref.data(), C_g.data(), N * S);

        // ── 输出 ──
        std::ostringstream ss_sub;
        if (sub_kib < 1024)
            ss_sub << sub_kib << " KiB";
        else
            ss_sub << std::fixed << std::setprecision(1) << (sub_kib / 1024.0) << " MiB";

        std::ostringstream ss_comp;
        ss_comp << std::fixed << std::setprecision(1) << cmin << "~" << cmax;

        std::ostringstream ss_total;
        ss_total << std::fixed << std::setprecision(1) << mean_total
                 << "±" << std::setprecision(1) << sd_total;

        // L1 miss rate 字符串
        std::string s_l1mr = "—";
        if (l1_miss_rate >= 0) {
            std::ostringstream ss; ss << std::fixed << std::setprecision(2) << l1_miss_rate << "%";
            s_l1mr = ss.str();
        }
        std::string s_l2mr = "—";
        if (l2_miss_rate >= 0) {
            std::ostringstream ss; ss << std::fixed << std::setprecision(2) << l2_miss_rate << "%";
            s_l2mr = ss.str();
        }

        std::cout << std::left
                  << std::setw(10) << ("G=" + std::to_string(Gi)).c_str()
                  << std::setw(8) << Gi
                  << std::setw(14) << ss_sub.str()
                  << std::setw(8) << cl
                  << std::setw(18) << ss_comp.str()
                  << std::setw(12) << std::fixed << std::setprecision(1) << avg_sync
                  << std::setw(14) << std::fixed << std::setprecision(1) << avg_scalar
                  << std::setw(14) << std::fixed << std::setprecision(1) << avg_sve
                  << std::setw(14) << std::fixed << std::setprecision(1) << prep_us
                  << std::setw(12) << s_l1mr
                  << std::setw(12) << s_l2mr
                  << std::setw(18) << ss_total.str()
                  << std::setw(12) << std::fixed << std::setprecision(1) << min_total
                  << std::setw(10) << std::fixed << std::setprecision(2) << gops_min
                  << std::setw(10) << (ok ? "OK" : "FAIL") << "\n";
    }
    std::cout << "\n";
}

// ====================== 实验7：单核 L1 小表微基准 ======================

void exp7_single_core_l1_lookup() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "实验7: 单核 L1 小表微基准 (SVE 归约)\n";
    std::cout << std::string(70, '=') << "\n";

    constexpr int TABLE_B = 256;
    constexpr int MAX_LOOKUPS = 128;      // 最大查表次数（table_a=2 时）
    constexpr int TABLE_A_VALS[] = {2, 4, 8, 16, 32};
    constexpr int WARMUP = 2000;
    constexpr int ITERS = 20000;

    // 输出表头
    std::cout << std::left
              << std::setw(16) << "表维度"
              << std::setw(14) << "表大小(KiB)"
              << std::setw(14) << "查表次数"
              << std::setw(16) << "平均耗时(μs)"
              << std::setw(14) << "总查表(ns)"
              << std::setw(14) << "每次查表(ns)"
              << std::setw(14) << "L1-miss%"
              << std::setw(14) << "标量(μs)"
              << std::setw(14) << "SVE累加(μs)"
              << std::setw(14) << "归约(μs)"
              << "\n" << std::string(140, '-') << "\n";

    alignas(64) float local_vals[MAX_LOOKUPS];  // 预分配最大空间

    for (int table_a : TABLE_A_VALS) {
        int entries = table_a * TABLE_B;
        int kib = entries * 2 / 1024;   // BF16 2B/entry
        int n_lookups = MAX_LOOKUPS / (table_a / 2);  // 128, 64, 32, 16, 8

        // 构建 BF16 表
        std::vector<uint16_t> sub(entries);
        for (int a = 0; a < table_a; ++a)
            for (int b = 0; b < TABLE_B; ++b)
                sub[a * TABLE_B + b] = float_to_bf16(sinf(a * 0.1f) * cosf(b * 0.1f));

        // 生成 (a, b) 对，a ∈ [0, table_a-1]
        std::vector<uint8_t> a_vals(n_lookups), b_vals(n_lookups);
        fill_random(a_vals.data(), n_lookups);
        for (auto &v : a_vals) v %= table_a;
        fill_random(b_vals.data(), n_lookups);

        volatile float sink = 0;

        // Warmup
        for (int w = 0; w < WARMUP; ++w) {
            for (int i = 0; i < n_lookups; ++i) {
                int idx = a_vals[i] * TABLE_B + b_vals[i];
                uint32_t bits = (uint32_t)sub[idx] << 16;
                memcpy(&local_vals[i], &bits, 4);
            }
            svfloat32_t acc = svdup_n_f32(0.0f);
            int t = 0;
            svbool_t pg = svwhilelt_b32(t, n_lookups);
            while (svptest_any(svptrue_b32(), pg)) {
                acc = svadd_f32_m(pg, acc, svld1_f32(pg, local_vals + t));
                t += svcntw();
                pg = svwhilelt_b32(t, n_lookups);
            }
            sink = svaddv_f32(svptrue_b32(), acc);
        }

        // PMU 计数器（累计 ITERS 次迭代）
        PerfCounter pc_l1_acc(ArmPmu::L1D_CACHE);
        PerfCounter pc_l1_miss(ArmPmu::L1D_CACHE_REFILL);
        bool perf_ok = pc_l1_acc.ok() && pc_l1_miss.ok();
        if (perf_ok) {
            pc_l1_acc.reset(); pc_l1_acc.enable();
            pc_l1_miss.reset(); pc_l1_miss.enable();
        }

        // 正式测量：20000 次，分阶段统计时间
        double sum_total = 0, sum_scalar = 0, sum_sve_acc = 0, sum_sve_reduce = 0;
        for (int iter = 0; iter < ITERS; ++iter) {
            auto t0 = high_resolution_clock::now();

            // 阶段1: 标量查表 + BF16→float
            for (int i = 0; i < n_lookups; ++i) {
                int idx = a_vals[i] * TABLE_B + b_vals[i];
                uint32_t bits = (uint32_t)sub[idx] << 16;
                memcpy(&local_vals[i], &bits, 4);
            }
            auto t1 = high_resolution_clock::now();

            // 阶段2: SVE 向量化累加 (svadd_f32_m)
            svfloat32_t acc = svdup_n_f32(0.0f);
            int t = 0;
            svbool_t pg = svwhilelt_b32(t, n_lookups);
            while (svptest_any(svptrue_b32(), pg)) {
                acc = svadd_f32_m(pg, acc, svld1_f32(pg, local_vals + t));
                t += svcntw();
                pg = svwhilelt_b32(t, n_lookups);
            }
            auto t2 = high_resolution_clock::now();

            // 阶段3: SVE 归约 (svaddv_f32)
            sink = svaddv_f32(svptrue_b32(), acc);
            auto t3 = high_resolution_clock::now();

            sum_scalar   += duration_cast<nanoseconds>(t1 - t0).count() / 1000.0;
            sum_sve_acc  += duration_cast<nanoseconds>(t2 - t1).count() / 1000.0;
            sum_sve_reduce += duration_cast<nanoseconds>(t3 - t2).count() / 1000.0;
            sum_total    += duration_cast<nanoseconds>(t3 - t0).count() / 1000.0;
        }

        if (perf_ok) {
            pc_l1_acc.disable();
            pc_l1_miss.disable();
        }

        double avg_us = sum_total / ITERS;
        double avg_ns_total = avg_us * 1000;
        double avg_ns_each = avg_ns_total / n_lookups;
        double avg_scalar_us = sum_scalar / ITERS;
        double avg_sve_acc_us = sum_sve_acc / ITERS;
        double avg_sve_reduce_us = sum_sve_reduce / ITERS;

        // L1-miss%
        std::string s_l1mr = "—";
        if (perf_ok) {
            uint64_t l1_a = pc_l1_acc.read();
            uint64_t l1_m = pc_l1_miss.read();
            if (l1_a > 0) {
                double rate = 100.0 * l1_m / l1_a;
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(2) << rate << "%";
                s_l1mr = ss.str();
            }
        }

        std::cout << std::left
                  << std::setw(16) << (std::to_string(table_a) + "×256").c_str()
                  << std::setw(14) << kib
                  << std::setw(14) << n_lookups
                  << std::setw(16) << std::fixed << std::setprecision(4) << avg_us
                  << std::setw(14) << std::fixed << std::setprecision(2) << avg_ns_total
                  << std::setw(14) << std::fixed << std::setprecision(2) << avg_ns_each
                  << std::setw(14) << s_l1mr
                  << std::setw(14) << std::fixed << std::setprecision(4) << avg_scalar_us
                  << std::setw(14) << std::fixed << std::setprecision(4) << avg_sve_acc_us
                  << std::setw(14) << std::fixed << std::setprecision(4) << avg_sve_reduce_us
                  << "\n";
    }
    std::cout << std::string(100, '-') << "\n";
}

// ====================== 实验8：两级查表（float 子表 + 行并行，无 atomic） ======================

/** 从完整 LUT 构建 G 个 float 细表（4B/entry，直接复制，无需 BF16 转换） */
static void build_float_fine_tables(int G,
                                     std::vector<std::vector<float>> &fine_tables,
                                     const float *lut) {
    int step = 256 / G;
    fine_tables.resize(G);
    for (int g = 0; g < G; ++g) {
        fine_tables[g].resize(step * 256);
        std::copy(lut + g * step * 256, lut + (g + 1) * step * 256,
                  fine_tables[g].begin());
    }
}

/**
 * 两级查表核函数：float 子表 + 行并行
 *
 * 对每个 (i,j)，遍历 G 组。每组内标量收集 B_T + 计算子表索引，
 * SVE gather 从 float 子表加载值并向量化累加。无 atomic。
 */
void lookup_two_level_omp(const GroupData &data,
                           const std::vector<std::vector<float>> &fine_tables,
                           const uint8_t *B_T, float *C,
                           int N, int S, int L,
                           float *core_times, float &sync_time,
                           float *scalar_times = nullptr,
                           float *sve_times = nullptr) {
    int G = data.G, step = data.step;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();

        // 静态行分配（模拟 schedule(static)）
        int rows_per = N / nt;
        int rem = N % nt;
        int i_start = tid * rows_per + std::min(tid, rem);
        int i_end = i_start + rows_per + (tid < rem ? 1 : 0);

        double t_scalar = 0, t_sve = 0;
        double t0 = omp_get_wtime();

        for (int i = i_start; i < i_end; ++i) {
            for (int j = 0; j < S; ++j) {
                float sum = 0.0f;
                for (int g = 0; g < G; ++g) {
                    int cnt = data.row_count[i * G + g];
                    if (cnt == 0) continue;

                    int start = data.row_start[i * G + g];
                    const uint8_t *a_ptr = data.A_grouped.data() + start;
                    const int *k_ptr = data.k_indices.data() + start;
                    const uint8_t *b_row = B_T + j * L;
                    const float *fine = fine_tables[g].data();
                    int base = g * step;

                    // 阶段1：标量收集 B_T + 计算子表索引
                    double ts0 = omp_get_wtime();
                    alignas(64) uint32_t idx_buf[512];
                    for (int t = 0; t < cnt; ++t)
                        idx_buf[t] = (uint32_t)(a_ptr[t] - base) * 256u + b_row[k_ptr[t]];
                    t_scalar += omp_get_wtime() - ts0;

                    // 阶段2：SVE gather + 累加 + 归约
                    double ts1 = omp_get_wtime();
                    svfloat32_t acc = svdup_n_f32(0.0f);
                    int t = 0;
                    svbool_t pg = svwhilelt_b32(t, cnt);
                    while (svptest_any(svptrue_b32(), pg)) {
                        acc = svadd_f32_m(pg, acc,
                            svld1_gather_u32index_f32(pg, fine, svld1_u32(pg, idx_buf + t)));
                        t += svcntw();
                        pg = svwhilelt_b32(t, cnt);
                    }
                    sum += svaddv_f32(svptrue_b32(), acc);
                    t_sve += omp_get_wtime() - ts1;
                }
                C[i * S + j] = sum;
            }
        }

        double t1 = omp_get_wtime();
        core_times[tid] = (t1 - t0) * 1e6;
        if (scalar_times) scalar_times[tid] = t_scalar * 1e6;
        if (sve_times)    sve_times[tid]    = t_sve * 1e6;

        #pragma omp barrier
        #pragma omp master
        sync_time = (omp_get_wtime() - t1) * 1e6;
    }
}

void exp8_two_level_lookup(const float *table, const uint8_t *A,
                            const uint8_t *B_T, const float *ref,
                            int N, int S, int L, int num_threads) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "实验8: 两级查表优化 (float 子表 + 行并行, " << num_threads << " 核)\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "矩阵: " << N << "×" << L << " * " << S << "×" << L << "^T\n";
    std::cout << "子表格式: float (4B), 无 atomic, 行并行\n";
    std::cout << "Warmup=" << 100 << ", 采样=" << 10000 << "\n\n";

    auto gops = [&](double us) { return double(N) * S * L / us / 1e3; };

    // ——— 基线：SVE + OpenMP 无分组 ———
    omp_set_num_threads(num_threads);
    std::vector<float> C_bl(N * S);
    double base_us = bench([&]() {
        memset(C_bl.data(), 0, N * S * sizeof(float));
        lookup_sve_omp(table, A, B_T, C_bl.data(), N, S, L);
    });
    bool base_ok = verify(ref, C_bl.data(), N * S);

    // ——— 输出表头（与实验6相同的 15 列） ———
    std::cout << std::left
              << std::setw(10) << "方案"
              << std::setw(8) << "线程"
              << std::setw(14) << "子表"
              << std::setw(8) << "级别"
              << std::setw(18) << "计算(核min~max)"
              << std::setw(12) << "同步"
              << std::setw(14) << "标量相位"
              << std::setw(14) << "SVE相位"
              << std::setw(14) << "预处理"
              << std::setw(12) << "L1-miss%"
              << std::setw(12) << "L2-miss%"
              << std::setw(18) << "总耗时mean±sd"
              << std::setw(12) << "总耗时min"
              << std::setw(10) << "GOP/s"
              << std::setw(10) << "正确"
              << "\n" << std::string(175, '-') << "\n";

    // 基线行（per-core 指标不可用，填 "—"）
    std::cout << std::left
              << std::setw(10) << "基线SVE"
              << std::setw(8) << num_threads
              << std::setw(14) << "256 KiB"
              << std::setw(8) << "L2"
              << std::setw(18) << "—"
              << std::setw(12) << "—"
              << std::setw(14) << "—"
              << std::setw(14) << "—"
              << std::setw(14) << "—"
              << std::setw(12) << "—"
              << std::setw(12) << "—"
              << std::setw(18) << std::fixed << std::setprecision(1) << base_us
              << std::setw(12) << "—"
              << std::setw(10) << std::fixed << std::setprecision(2) << gops(base_us)
              << std::setw(10) << (base_ok ? "OK" : "FAIL") << "\n";

    // ——— 扫描 G ———
    constexpr int G_VALS[] = {1, 2, 4, 8, 16, 32, 64};
    constexpr int WARMUP = 100;
    constexpr int ITERS = 10000;

    for (int Gi : G_VALS) {
        if (Gi > omp_get_max_threads()) continue;

        // ── 预处理（计时） ──
        double t_prep = omp_get_wtime();
        GroupData gd = preprocess_groups(A, N, L, Gi);
        std::vector<std::vector<float>> fine_tables;
        build_float_fine_tables(Gi, fine_tables, table);
        double prep_us = (omp_get_wtime() - t_prep) * 1e6;

        int sub_kib = (256 / Gi) * 256 * 4 / 1024;  // float 4B
        const char *cl = cache_level(sub_kib, L1D_KiB, L2_KiB);

        // ── Perf 计数器（打开失败则降级） ──
        PerfCounter pc_l1_acc(ArmPmu::L1D_CACHE);
        PerfCounter pc_l1_miss(ArmPmu::L1D_CACHE_REFILL);
        PerfCounter pc_l2_acc(ArmPmu::L2D_CACHE);
        PerfCounter pc_l2_miss(ArmPmu::L2D_CACHE_REFILL);
        bool perf_ok = pc_l1_acc.ok() && pc_l1_miss.ok();

        // ── 复用缓冲区 ──
        std::vector<float> C_g(N * S, 0);
        std::vector<float> core_t(num_threads);
        std::vector<float> scalar_t(num_threads), sve_t(num_threads);
        float sync_t = 0;

        // ── Warmup ──
        for (int w = 0; w < WARMUP; ++w) {
            std::fill(C_g.begin(), C_g.end(), 0);
            lookup_two_level_omp(gd, fine_tables, B_T, C_g.data(), N, S, L,
                                 core_t.data(), sync_t);
        }

        // ── 测量 ──
        // 在线统计：总耗时
        double sum_total = 0, sum_total2 = 0;
        double min_total = 1e18, max_total = 0;
        // 同步时间
        double sum_sync = 0;
        // 阶段耗时（累计后取平均）
        double sum_scalar = 0, sum_sve = 0;
        // 最佳单次（用于输出 compute min~max）
        double best_total = 1e18;
        float best_sync = 0;
        std::vector<float> best_core(num_threads);

        // 启动 perf 计数器（从第一次测量开始累计）
        if (perf_ok) {
            pc_l1_acc.reset(); pc_l1_acc.enable();
            pc_l1_miss.reset(); pc_l1_miss.enable();
            if (pc_l2_acc.ok()) { pc_l2_acc.reset(); pc_l2_acc.enable(); }
            if (pc_l2_miss.ok()) { pc_l2_miss.reset(); pc_l2_miss.enable(); }
        }

        for (int iter = 0; iter < ITERS; ++iter) {
            std::fill(C_g.begin(), C_g.end(), 0);
            std::fill(core_t.begin(), core_t.end(), 0);
            std::fill(scalar_t.begin(), scalar_t.end(), 0);
            std::fill(sve_t.begin(), sve_t.end(), 0);
            sync_t = 0;

            lookup_two_level_omp(gd, fine_tables, B_T, C_g.data(), N, S, L,
                                 core_t.data(), sync_t,
                                 scalar_t.data(), sve_t.data());

            // 累加阶段耗时（取每核 max，因为最慢核决定总时间）
            double max_scalar = *std::max_element(scalar_t.begin(), scalar_t.end());
            double max_sve = *std::max_element(sve_t.begin(), sve_t.end());
            sum_scalar += max_scalar;
            sum_sve += max_sve;

            double total = *std::max_element(core_t.begin(), core_t.end()) + sync_t;

            // 更新在线统计
            sum_total += total;
            sum_total2 += total * total;
            if (total < min_total) min_total = total;
            if (total > max_total) max_total = total;
            sum_sync += sync_t;

            // 记录最优单次（用于输出计算核时间）
            if (total < best_total) {
                best_total = total;
                best_sync = sync_t;
                best_core = core_t;
            }
        }

        // 停止 perf 计数器
        if (perf_ok) {
            pc_l1_acc.disable();
            pc_l1_miss.disable();
            if (pc_l2_acc.ok()) pc_l2_acc.disable();
            if (pc_l2_miss.ok()) pc_l2_miss.disable();
        }

        // ── 统计计算 ──
        double mean_total = sum_total / ITERS;
        double var_total = (sum_total2 - sum_total * mean_total) / (ITERS - 1);
        double sd_total = std::sqrt(std::max(0.0, var_total));
        double avg_sync = sum_sync / ITERS;
        double avg_scalar = sum_scalar / ITERS;
        double avg_sve = sum_sve / ITERS;

        float cmin = *std::min_element(best_core.begin(), best_core.end());
        float cmax = *std::max_element(best_core.begin(), best_core.end());
        double gops_min = gops(min_total);

        // ── Cache miss rate ──
        double l1_miss_rate = -1, l2_miss_rate = -1;
        if (perf_ok) {
            uint64_t l1_a = pc_l1_acc.read();
            uint64_t l1_m = pc_l1_miss.read();
            if (l1_a > 0) l1_miss_rate = 100.0 * l1_m / l1_a;

            if (pc_l2_acc.ok() && pc_l2_miss.ok()) {
                uint64_t l2_a = pc_l2_acc.read();
                uint64_t l2_m = pc_l2_miss.read();
                if (l2_a > 0) l2_miss_rate = 100.0 * l2_m / l2_a;
            }
        }

        bool ok = verify(ref, C_g.data(), N * S);

        // ── 输出 ──
        std::ostringstream ss_sub;
        ss_sub << sub_kib << " KiB";

        std::ostringstream ss_comp;
        ss_comp << std::fixed << std::setprecision(1) << cmin << "~" << cmax;

        std::ostringstream ss_total;
        ss_total << std::fixed << std::setprecision(1) << mean_total
                 << "±" << std::setprecision(1) << sd_total;

        // L1 miss rate 字符串
        std::string s_l1mr = "—";
        if (l1_miss_rate >= 0) {
            std::ostringstream ss; ss << std::fixed << std::setprecision(2) << l1_miss_rate << "%";
            s_l1mr = ss.str();
        }
        std::string s_l2mr = "—";
        if (l2_miss_rate >= 0) {
            std::ostringstream ss; ss << std::fixed << std::setprecision(2) << l2_miss_rate << "%";
            s_l2mr = ss.str();
        }

        std::cout << std::left
                  << std::setw(10) << ("TL-G=" + std::to_string(Gi)).c_str()
                  << std::setw(8) << num_threads
                  << std::setw(14) << ss_sub.str()
                  << std::setw(8) << cl
                  << std::setw(18) << ss_comp.str()
                  << std::setw(12) << std::fixed << std::setprecision(1) << avg_sync
                  << std::setw(14) << std::fixed << std::setprecision(1) << avg_scalar
                  << std::setw(14) << std::fixed << std::setprecision(1) << avg_sve
                  << std::setw(14) << std::fixed << std::setprecision(1) << prep_us
                  << std::setw(12) << s_l1mr
                  << std::setw(12) << s_l2mr
                  << std::setw(18) << ss_total.str()
                  << std::setw(12) << std::fixed << std::setprecision(1) << min_total
                  << std::setw(10) << std::fixed << std::setprecision(2) << gops_min
                  << std::setw(10) << (ok ? "OK" : "FAIL") << "\n";
    }
    std::cout << "\n";
}

// ====================== 实验9：矩阵加载对查表微基准的影响 ======================

void exp9_matrix_load_microbench() {
    constexpr int N = 128, S = 328, L = 512;
    constexpr int TABLE_DIMS[] = {256, 128, 64, 32, 16};
    constexpr int WARMUP = 20;
    constexpr int ITERS = 100;

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "实验9: 标量 vs SVE gather — 不同 LUT 维度对比 (单核)\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "矩阵: " << N << "×" << L << " × " << S << "×" << L << "^T\n";
    std::cout << "LUT 格式: float (4B), 单核\n";
    std::cout << "标量: lookup_scalar (顺序加载)\n";
    std::cout << "SVE:  gather_lookup (svld1_gather_u32index_f32, _z 归约)\n";
    std::cout << "Warmup=" << WARMUP << ", 采样=" << ITERS << "\n\n";

    auto bench_us = [&](auto fn) {
        for (int w = 0; w < WARMUP; ++w) fn();
        double sum = 0;
        for (int t = 0; t < ITERS; ++t) {
            auto t0 = high_resolution_clock::now();
            fn();
            auto t1 = high_resolution_clock::now();
            sum += duration_cast<nanoseconds>(t1 - t0).count() / 1000.0;
        }
        return sum / ITERS;
    };

    // 输出表头
    std::cout << std::left
              << std::setw(14) << "LUT 维度"
              << std::setw(14) << "LUT 大小"
              << std::setw(16) << "Cache 层级"
              << std::setw(18) << "标量(μs)"
              << std::setw(18) << "SVE gather(μs)"
              << std::setw(14) << "加速比"
              << "\n" << std::string(95, '-') << "\n";

    for (int dim : TABLE_DIMS) {
        int lut_kib = dim * dim * 4 / 1024;

        // 生成 LUT: dim × dim float
        std::vector<float> lut(dim * dim);
        for (int a = 0; a < dim; ++a)
            for (int b = 0; b < dim; ++b)
                lut[a * dim + b] = sinf(a * 0.1f) * cosf(b * 0.1f);

        // 生成 A, B_T，值约束到 [0, dim-1]
        std::vector<uint8_t> A(N * L), B_T(S * L);
        for (auto &v : A) v = rand() % dim;
        for (auto &v : B_T) v = rand() % dim;

        std::vector<float> C(N * S);

        // ——— 标量版本 ———
        double t_scalar = bench_us([&]() {
            for (int i = 0; i < N; ++i) {
                const uint8_t *rowA = A.data() + i * L;
                for (int j = 0; j < S; ++j) {
                    const uint8_t *rowB = B_T.data() + j * L;
                    float sum = 0.0f;
                    for (int k = 0; k < L; ++k)
                        sum += lut[rowA[k] * dim + rowB[k]];
                    C[i * S + j] = sum;
                }
            }
        });

        // ——— SVE gather 版本 ———
        double t_sve = bench_us([&]() {
            for (int i = 0; i < N; ++i) {
                const uint8_t *rowA = A.data() + i * L;
                for (int j = 0; j < S; ++j) {
                    const uint8_t *rowB = B_T.data() + j * L;
                    svfloat32_t acc = svdup_n_f32(0.0f);
                    int k = 0;
                    svbool_t pg = svwhilelt_b32(k, L);
                    while (svptest_any(svptrue_b32(), pg)) {
                        svuint32_t idxA = svld1ub_u32(pg, &rowA[k]);
                        svuint32_t idxB = svld1ub_u32(pg, &rowB[k]);
                        // idxA * dim + idxB（dim 在编译期常量传播后为 immed）
                        svuint32_t idx = svadd_u32_z(pg,
                            svmul_n_u32_z(pg, idxA, dim), idxB);
                        acc = svadd_f32_z(pg, acc,
                            svld1_gather_u32index_f32(pg, lut.data(), idx));
                        k += svcntw();
                        pg = svwhilelt_b32(k, L);
                    }
                    C[i * S + j] = svaddv_f32(svptrue_b32(), acc);
                }
            }
        });

        const char *cache_lvl = (lut_kib <= 64) ? "L1" : "L2";
        double speedup = t_scalar / t_sve;

        // LUT 大小字符串
        std::string s_size;
        if (lut_kib < 1024)
            s_size = std::to_string(lut_kib) + " KiB";
        else
            s_size = std::to_string(lut_kib / 1024) + " MiB";

        std::cout << std::left
                  << std::setw(14) << (std::to_string(dim) + "×" + std::to_string(dim))
                  << std::setw(14) << s_size
                  << std::setw(16) << cache_lvl
                  << std::setw(18) << std::fixed << std::setprecision(1) << t_scalar
                  << std::setw(18) << std::fixed << std::setprecision(1) << t_sve
                  << std::setw(14) << std::fixed << std::setprecision(2) << speedup << "×"
                  << "\n";
    }
    std::cout << "\n";
}

// ====================== 实验10：纯查表微基准（预计算索引，无索引计算开销） ======================

/**
 * 实验10：在实验7的基础上，将索引对 (a,b) 改为预计算好的索引数组。
 *
 * 与实验7的区别：
 *   实验7标量相位： idx = a_vals[i] * TABLE_B + b_vals[i]  （含乘+加索引计算）
 *   实验10标量相位： local_vals[i] = sub[indices[i]]        （纯查表，无索引计算）
 *
 * 对比实验7即可测得索引计算开销。
 *
 * @param n_lookups  查表次数（默认 0 表示按实验7规则自动计算）
 */
void exp10_pure_lookup_microbench(int n_lookups_user = 0) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "实验10: 纯查表微基准 — 预计算索引 (无索引计算开销)\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "数据格式: float (4B/entry), 无 BF16 转换\n";
    std::cout << "与实验7区别: 索引预计算 + float 直接读取，标量相位仅含查表\n";
    std::cout << "对比实验7的标量相位即可测得索引计算 ((a)*256+(b)) + BF16 转换的开销\n\n";

    constexpr int TABLE_B = 256;
    constexpr int MAX_LOOKUPS = 128;
    constexpr int TABLE_A_VALS[] = {2, 4, 8, 16, 32};
    constexpr int WARMUP = 2000;
    constexpr int ITERS = 20000;

    // 输出表头（与实验7格式一致）
    std::cout << std::left
              << std::setw(16) << "表维度"
              << std::setw(14) << "表大小(KiB)"
              << std::setw(14) << "查表次数"
              << std::setw(16) << "平均耗时(μs)"
              << std::setw(14) << "总查表(ns)"
              << std::setw(14) << "每次查表(ns)"
              << std::setw(14) << "L1-miss%"
              << std::setw(14) << "标量(μs)"
              << std::setw(14) << "SVE累加(μs)"
              << std::setw(14) << "归约(μs)"
              << std::setw(16) << "标量vs实验7"
              << "\n" << std::string(156, '-') << "\n";

    for (int table_a : TABLE_A_VALS) {
        int entries = table_a * TABLE_B;
        int kib = entries * 4 / 1024;   // float 4B/entry
        // 用户指定 n_lookups 则使用用户值，否则按实验7规则
        int n_lookups = (n_lookups_user > 0) ? n_lookups_user
                                             : (MAX_LOOKUPS / (table_a / 2));

        // 动态分配（用户可传入任意长度）
        std::vector<float> local_vals(n_lookups);
        std::vector<uint32_t> indices(n_lookups);

        // 构建 float 表
        std::vector<float> sub(entries);
        for (int a = 0; a < table_a; ++a)
            for (int b = 0; b < TABLE_B; ++b)
                sub[a * TABLE_B + b] = sinf(a * 0.1f) * cosf(b * 0.1f);

        // 预计算索引数组：indices[i] = rand() % entries
        for (int i = 0; i < n_lookups; ++i)
            indices[i] = rand() % entries;

        volatile float sink = 0;

        // Warmup
        for (int w = 0; w < WARMUP; ++w) {
            for (int i = 0; i < n_lookups; ++i)
                local_vals[i] = sub[indices[i]];
            svfloat32_t acc = svdup_n_f32(0.0f);
            int t = 0;
            svbool_t pg = svwhilelt_b32(t, n_lookups);
            while (svptest_any(svptrue_b32(), pg)) {
                acc = svadd_f32_m(pg, acc, svld1_f32(pg, local_vals.data() + t));
                t += svcntw();
                pg = svwhilelt_b32(t, n_lookups);
            }
            sink = svaddv_f32(svptrue_b32(), acc);
        }

        // PMU 计数器
        PerfCounter pc_l1_acc(ArmPmu::L1D_CACHE);
        PerfCounter pc_l1_miss(ArmPmu::L1D_CACHE_REFILL);
        bool perf_ok = pc_l1_acc.ok() && pc_l1_miss.ok();
        if (perf_ok) {
            pc_l1_acc.reset(); pc_l1_acc.enable();
            pc_l1_miss.reset(); pc_l1_miss.enable();
        }

        // 正式测量：分阶段计时
        double sum_total = 0, sum_scalar = 0, sum_sve_acc = 0, sum_sve_reduce = 0;
        for (int iter = 0; iter < ITERS; ++iter) {
            auto t0 = high_resolution_clock::now();

            // 阶段1: 纯查表（无索引计算，float 直接读取）
            for (int i = 0; i < n_lookups; ++i)
                local_vals[i] = sub[indices[i]];
            auto t1 = high_resolution_clock::now();

            // 阶段2: SVE 向量化累加
            svfloat32_t acc = svdup_n_f32(0.0f);
            int t = 0;
            svbool_t pg = svwhilelt_b32(t, n_lookups);
            while (svptest_any(svptrue_b32(), pg)) {
                acc = svadd_f32_m(pg, acc, svld1_f32(pg, local_vals.data() + t));
                t += svcntw();
                pg = svwhilelt_b32(t, n_lookups);
            }
            auto t2 = high_resolution_clock::now();

            // 阶段3: SVE 归约
            sink = svaddv_f32(svptrue_b32(), acc);
            auto t3 = high_resolution_clock::now();

            sum_scalar   += duration_cast<nanoseconds>(t1 - t0).count() / 1000.0;
            sum_sve_acc  += duration_cast<nanoseconds>(t2 - t1).count() / 1000.0;
            sum_sve_reduce += duration_cast<nanoseconds>(t3 - t2).count() / 1000.0;
            sum_total    += duration_cast<nanoseconds>(t3 - t0).count() / 1000.0;
        }

        if (perf_ok) {
            pc_l1_acc.disable();
            pc_l1_miss.disable();
        }

        double avg_us = sum_total / ITERS;
        double avg_ns_total = avg_us * 1000;
        double avg_ns_each = avg_ns_total / n_lookups;
        double avg_scalar_us = sum_scalar / ITERS;
        double avg_sve_acc_us = sum_sve_acc / ITERS;
        double avg_sve_reduce_us = sum_sve_reduce / ITERS;

        // L1-miss%
        std::string s_l1mr = "—";
        if (perf_ok) {
            uint64_t l1_a = pc_l1_acc.read();
            uint64_t l1_m = pc_l1_miss.read();
            if (l1_a > 0) {
                double rate = 100.0 * l1_m / l1_a;
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(2) << rate << "%";
                s_l1mr = ss.str();
            }
        }

        std::cout << std::left
                  << std::setw(16) << (std::to_string(table_a) + "×256").c_str()
                  << std::setw(14) << kib
                  << std::setw(14) << n_lookups
                  << std::setw(16) << std::fixed << std::setprecision(4) << avg_us
                  << std::setw(14) << std::fixed << std::setprecision(2) << avg_ns_total
                  << std::setw(14) << std::fixed << std::setprecision(2) << avg_ns_each
                  << std::setw(14) << s_l1mr
                  << std::setw(14) << std::fixed << std::setprecision(4) << avg_scalar_us
                  << std::setw(14) << std::fixed << std::setprecision(4) << avg_sve_acc_us
                  << std::setw(14) << std::fixed << std::setprecision(4) << avg_sve_reduce_us
                  << std::setw(16) << "—"  // 手动对比
                  << "\n";
    }
    std::cout << std::string(156, '-') << "\n";
    std::cout << "注: \"标量vs实验7\" 列需手动对比实验7输出。\n";
    std::cout << "    实验10标量相位 = float 纯查表 | 实验7标量相位 = BF16 查表 + 索引计算 + BF16→float\n";
    std::cout << "    两者差值 = 索引计算 (a*256+b) + BF16 转换开销\n";

    // 如有用户指定 n_lookups 参数，打印提示
    if (n_lookups_user > 0)
        std::cout << "    使用外部传入的查表次数: " << n_lookups_user << "\n";
    std::cout << "\n";
}


// ====================== 实验11：SVE gather 查表微基准（预计算索引） ======================

/**
 * 实验11：在实验10的基础上，将标量查表替换为 SVE gather 查表。
 *
 * 与实验10的区别：
 *   实验10标量相位： local_vals[i] = sub[indices[i]]              （标量逐元素）
 *   实验11标量相位： SVE gather → svst1 写入 local_vals            （向量化 gather）
 *
 * Phase 1 直接对比可测得 SVE gather 相对标量加载的加速效果。
 */
void exp11_sve_gather_microbench(int n_lookups_user = 0) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "实验11: SVE gather 查表微基准 — 预计算索引\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "数据格式: float (4B/entry), SVE gather 查表 (svld1_gather_u32index_f32)\n";
    std::cout << "与实验10区别: 标量逐元素查表 → SVE gather 向量化查表\n";
    std::cout << "对比实验10的标量相位即可测得 SVE gather 相对标量加载的加速比\n\n";

    constexpr int TABLE_B = 256;
    constexpr int MAX_LOOKUPS = 128;
    constexpr int TABLE_A_VALS[] = {2, 4, 8, 16, 32};
    constexpr int WARMUP = 2000;
    constexpr int ITERS = 20000;

    // 输出表头（与实验10格式一致，"标量" → "SVE gather"）
    std::cout << std::left
              << std::setw(16) << "表维度"
              << std::setw(14) << "表大小(KiB)"
              << std::setw(14) << "查表次数"
              << std::setw(16) << "平均耗时(μs)"
              << std::setw(14) << "总查表(ns)"
              << std::setw(14) << "每次查表(ns)"
              << std::setw(14) << "L1-miss%"
              << std::setw(14) << "SVE gather(μs)"
              << std::setw(14) << "SVE累加(μs)"
              << std::setw(14) << "归约(μs)"
              << std::setw(16) << "vs实验10标量"
              << "\n" << std::string(156, '-') << "\n";

    for (int table_a : TABLE_A_VALS) {
        int entries = table_a * TABLE_B;
        int kib = entries * 4 / 1024;
        int n_lookups = (n_lookups_user > 0) ? n_lookups_user
                                             : (MAX_LOOKUPS / (table_a / 2));

        // 构建 float 表
        std::vector<float> sub(entries);
        for (int a = 0; a < table_a; ++a)
            for (int b = 0; b < TABLE_B; ++b)
                sub[a * TABLE_B + b] = sinf(a * 0.1f) * cosf(b * 0.1f);

        // 预计算索引
        std::vector<uint32_t> indices(n_lookups);
        for (int i = 0; i < n_lookups; ++i)
            indices[i] = rand() % entries;

        // local_vals 供 Phase 2 (SVE累加) 使用，与实验10结构一致
        std::vector<float> local_vals(n_lookups);

        volatile float sink = 0;

        // Warmup
        for (int w = 0; w < WARMUP; ++w) {
            int t = 0;
            svbool_t pg = svwhilelt_b32(t, n_lookups);
            while (svptest_any(svptrue_b32(), pg)) {
                svuint32_t idx_vec = svld1_u32(pg, indices.data() + t);
                svfloat32_t gathered = svld1_gather_u32index_f32(pg, sub.data(), idx_vec);
                svst1_f32(pg, local_vals.data() + t, gathered);
                t += svcntw();
                pg = svwhilelt_b32(t, n_lookups);
            }
            svfloat32_t acc = svdup_n_f32(0.0f);
            t = 0;
            pg = svwhilelt_b32(t, n_lookups);
            while (svptest_any(svptrue_b32(), pg)) {
                acc = svadd_f32_m(pg, acc, svld1_f32(pg, local_vals.data() + t));
                t += svcntw();
                pg = svwhilelt_b32(t, n_lookups);
            }
            sink = svaddv_f32(svptrue_b32(), acc);
        }

        // PMU 计数器
        PerfCounter pc_l1_acc(ArmPmu::L1D_CACHE);
        PerfCounter pc_l1_miss(ArmPmu::L1D_CACHE_REFILL);
        bool perf_ok = pc_l1_acc.ok() && pc_l1_miss.ok();
        if (perf_ok) {
            pc_l1_acc.reset(); pc_l1_acc.enable();
            pc_l1_miss.reset(); pc_l1_miss.enable();
        }

        // 正式测量：三阶段计时（与实验10结构一致）
        double sum_total = 0, sum_gather = 0, sum_sve_acc = 0, sum_sve_reduce = 0;
        for (int iter = 0; iter < ITERS; ++iter) {
            auto t0 = high_resolution_clock::now();

            // 阶段1: SVE gather 查表（替代实验10的标量逐元素加载）
            int t = 0;
            svbool_t pg = svwhilelt_b32(t, n_lookups);
            while (svptest_any(svptrue_b32(), pg)) {
                svuint32_t idx_vec = svld1_u32(pg, indices.data() + t);
                svfloat32_t gathered = svld1_gather_u32index_f32(pg, sub.data(), idx_vec);
                svst1_f32(pg, local_vals.data() + t, gathered);
                t += svcntw();
                pg = svwhilelt_b32(t, n_lookups);
            }
            auto t1 = high_resolution_clock::now();

            // 阶段2: SVE 向量化累加
            svfloat32_t acc = svdup_n_f32(0.0f);
            t = 0;
            pg = svwhilelt_b32(t, n_lookups);
            while (svptest_any(svptrue_b32(), pg)) {
                acc = svadd_f32_m(pg, acc, svld1_f32(pg, local_vals.data() + t));
                t += svcntw();
                pg = svwhilelt_b32(t, n_lookups);
            }
            auto t2 = high_resolution_clock::now();

            // 阶段3: SVE 归约
            sink = svaddv_f32(svptrue_b32(), acc);
            auto t3 = high_resolution_clock::now();

            sum_gather    += duration_cast<nanoseconds>(t1 - t0).count() / 1000.0;
            sum_sve_acc   += duration_cast<nanoseconds>(t2 - t1).count() / 1000.0;
            sum_sve_reduce += duration_cast<nanoseconds>(t3 - t2).count() / 1000.0;
            sum_total     += duration_cast<nanoseconds>(t3 - t0).count() / 1000.0;
        }

        if (perf_ok) {
            pc_l1_acc.disable();
            pc_l1_miss.disable();
        }

        double avg_us = sum_total / ITERS;
        double avg_ns_total = avg_us * 1000;
        double avg_ns_each = avg_ns_total / n_lookups;
        double avg_gather_us = sum_gather / ITERS;
        double avg_sve_acc_us = sum_sve_acc / ITERS;
        double avg_sve_reduce_us = sum_sve_reduce / ITERS;

        // L1-miss%
        std::string s_l1mr = "—";
        if (perf_ok) {
            uint64_t l1_a = pc_l1_acc.read();
            uint64_t l1_m = pc_l1_miss.read();
            if (l1_a > 0) {
                double rate = 100.0 * l1_m / l1_a;
                std::ostringstream ss;
                ss << std::fixed << std::setprecision(2) << rate << "%";
                s_l1mr = ss.str();
            }
        }

        std::cout << std::left
                  << std::setw(16) << (std::to_string(table_a) + "×256").c_str()
                  << std::setw(14) << kib
                  << std::setw(14) << n_lookups
                  << std::setw(16) << std::fixed << std::setprecision(4) << avg_us
                  << std::setw(14) << std::fixed << std::setprecision(2) << avg_ns_total
                  << std::setw(14) << std::fixed << std::setprecision(2) << avg_ns_each
                  << std::setw(14) << s_l1mr
                  << std::setw(14) << std::fixed << std::setprecision(4) << avg_gather_us
                  << std::setw(14) << std::fixed << std::setprecision(4) << avg_sve_acc_us
                  << std::setw(14) << std::fixed << std::setprecision(4) << avg_sve_reduce_us
                  << std::setw(16) << "—"  // 手动对比
                  << "\n";
    }
    std::cout << std::string(156, '-') << "\n";
    std::cout << "注: \"vs实验10标量\" 列需手动对比实验10输出。\n";
    std::cout << "    实验11 Phase 1 = SVE gather 查表 | 实验10 Phase 1 = 标量逐元素查表\n";
    std::cout << "    两者差值 = SVE gather 相对标量加载的加速/减速\n";
    if (n_lookups_user > 0)
        std::cout << "    使用外部传入的查表次数: " << n_lookups_user << "\n";
    std::cout << "\n";
}


// ====================== 实验12：标量 vs SVE 查表分阶段对比 ======================

/**
 * 实验12：以 lookup_scalar / lookup_sve 两个函数为基准，
 * 将每个函数拆分为"索引计算"和"查表"两个阶段分别计时。
 *
 * 标量分解：
 *   阶段1（索引计算）：idx = (rA[k] << 8) | rB[k]
 *   阶段2（查表累加）：sum += table[idx]
 *
 * SVE 分解：
 *   阶段1（索引计算）：svorr(svlsl(idxA,8), idxB)
 *   阶段2（查表累加）：svld1_gather_u32index_f32 + svadd_f32_z
 *   阶段3（归约）：   svaddv_f32
 */
void exp12_lookup_phase_bench() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "实验12: 标量 vs SVE 查表分阶段对比\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "对每个 K 长度，分别测量完整函数、索引计算阶段、查表阶段的耗时\n\n";

    constexpr int TABLE_B = 256;
    constexpr int LENGTHS[] = {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
    constexpr int WARMUP = 2000;
    constexpr int ITERS = 20000;
    constexpr int MAX_LEN = 65536;

    // 预生成随机数据
    std::vector<uint8_t> A(MAX_LEN), B_T(MAX_LEN);
    fill_random(A.data(), MAX_LEN);
    fill_random(B_T.data(), MAX_LEN);

    // 全 LUT（256×256 float）
    std::vector<float> table(TABLE_B * TABLE_B);
    for (int a = 0; a < TABLE_B; ++a)
        for (int b = 0; b < TABLE_B; ++b)
            table[a * TABLE_B + b] = sinf(a * 0.1f) * cosf(b * 0.1f);

    // 预分配索引缓冲区（堆上，避免栈溢出）
    std::vector<uint32_t> idx_buf(MAX_LEN);
    std::vector<uint32_t> sve_idx_buf(MAX_LEN);

    // 输出表头
    std::cout << std::left
              << std::setw(10) << "K"
              << std::setw(14) << "标量总(μs)"
              << std::setw(14) << "标量索引(μs)"
              << std::setw(14) << "标量查表(μs)"
              << std::setw(14) << "SVE总(μs)"
              << std::setw(14) << "SVE索引(μs)"
              << std::setw(14) << "SVE查表(μs)"
              << std::setw(14) << "SVE归约(μs)"
              << std::setw(12) << "总加速比"
              << "\n" << std::string(130, '-') << "\n";

    for (int L : LENGTHS) {
        // ====== 标量总耗时（完整 lookup_scalar，N=1, S=1） ======
        auto scalar_full = [&]() {
            float sum = 0.0f;
            const uint8_t *rA = A.data(), *rB = B_T.data();
            for (int k = 0; k < L; ++k)
                sum += table[(rA[k] << 8) | rB[k]];
            volatile float sink = sum;
            (void)sink;
        };

        for (int w = 0; w < WARMUP; ++w) scalar_full();

        double t_scalar_total = 0;
        for (int iter = 0; iter < ITERS; ++iter) {
            auto t0 = high_resolution_clock::now();
            scalar_full();
            auto t1 = high_resolution_clock::now();
            t_scalar_total += duration_cast<nanoseconds>(t1 - t0).count() / 1000.0;
        }
        t_scalar_total /= ITERS;

        // ====== 标量索引计算阶段 ======
        auto scalar_idx = [&]() {
            const uint8_t *rA = A.data(), *rB = B_T.data();
            for (int k = 0; k < L; ++k)
                idx_buf[k] = ((uint32_t)rA[k] << 8) | rB[k];
        };

        for (int w = 0; w < WARMUP; ++w) scalar_idx();

        double t_scalar_idx = 0;
        for (int iter = 0; iter < ITERS; ++iter) {
            auto t0 = high_resolution_clock::now();
            scalar_idx();
            auto t1 = high_resolution_clock::now();
            t_scalar_idx += duration_cast<nanoseconds>(t1 - t0).count() / 1000.0;
        }
        t_scalar_idx /= ITERS;

        // ====== 标量查表累加阶段 ======
        auto scalar_lu = [&]() {
            float sum = 0.0f;
            for (int k = 0; k < L; ++k)
                sum += table[idx_buf[k]];
            volatile float sink = sum;
            (void)sink;
        };

        for (int w = 0; w < WARMUP; ++w) scalar_lu();

        double t_scalar_lu = 0;
        for (int iter = 0; iter < ITERS; ++iter) {
            auto t0 = high_resolution_clock::now();
            scalar_lu();
            auto t1 = high_resolution_clock::now();
            t_scalar_lu += duration_cast<nanoseconds>(t1 - t0).count() / 1000.0;
        }
        t_scalar_lu /= ITERS;

        // ====== SVE 总耗时（完整 lookup_sve，N=1, S=1） ======
        auto sve_full = [&]() {
            svfloat32_t acc = svdup_n_f32(0.0f);
            const uint8_t *rA = A.data(), *rB = B_T.data();
            int k = 0;
            svbool_t pg = svwhilelt_b32(k, L);
            while (svptest_any(svptrue_b32(), pg)) {
                svuint32_t idxA = svld1ub_u32(pg, &rA[k]);
                svuint32_t idxB = svld1ub_u32(pg, &rB[k]);
                svuint32_t idx = svorr_u32_z(pg, svlsl_n_u32_z(pg, idxA, 8), idxB);
                acc = svadd_f32_z(pg, acc, svld1_gather_u32index_f32(pg, table.data(), idx));
                k += svcntw();
                pg = svwhilelt_b32(k, L);
            }
            volatile float sink = svaddv_f32(svptrue_b32(), acc);
            (void)sink;
        };

        for (int w = 0; w < WARMUP; ++w) sve_full();

        double t_sve_total = 0;
        for (int iter = 0; iter < ITERS; ++iter) {
            auto t0 = high_resolution_clock::now();
            sve_full();
            auto t1 = high_resolution_clock::now();
            t_sve_total += duration_cast<nanoseconds>(t1 - t0).count() / 1000.0;
        }
        t_sve_total /= ITERS;

        // ====== SVE 索引计算阶段 ======
        auto sve_idx = [&]() {
            const uint8_t *rA = A.data(), *rB = B_T.data();
            int k = 0;
            svbool_t pg = svwhilelt_b32(k, L);
            while (svptest_any(svptrue_b32(), pg)) {
                svuint32_t idxA = svld1ub_u32(pg, &rA[k]);
                svuint32_t idxB = svld1ub_u32(pg, &rB[k]);
                svuint32_t idx = svorr_u32_z(pg, svlsl_n_u32_z(pg, idxA, 8), idxB);
                svst1_u32(pg, &sve_idx_buf[k], idx);
                k += svcntw();
                pg = svwhilelt_b32(k, L);
            }
        };

        for (int w = 0; w < WARMUP; ++w) sve_idx();

        double t_sve_idx = 0;
        for (int iter = 0; iter < ITERS; ++iter) {
            auto t0 = high_resolution_clock::now();
            sve_idx();
            auto t1 = high_resolution_clock::now();
            t_sve_idx += duration_cast<nanoseconds>(t1 - t0).count() / 1000.0;
        }
        t_sve_idx /= ITERS;

        // ====== SVE 查表累加阶段（使用预计算索引） ======
        auto sve_lu = [&]() {
            svfloat32_t acc = svdup_n_f32(0.0f);
            int k = 0;
            svbool_t pg = svwhilelt_b32(k, L);
            while (svptest_any(svptrue_b32(), pg)) {
                acc = svadd_f32_z(pg, acc,
                    svld1_gather_u32index_f32(pg, table.data(),
                        svld1_u32(pg, &sve_idx_buf[k])));
                k += svcntw();
                pg = svwhilelt_b32(k, L);
            }
            volatile float sink = svaddv_f32(svptrue_b32(), acc);
            (void)sink;
        };

        for (int w = 0; w < WARMUP; ++w) sve_lu();

        // 注意：sve_lu 内部含归约，单独再测归约阶段
        // 但归约在 sve_lu 中只执行一次，影响极小。用 sve_reduce 单独测量更准确。

        double t_sve_lu = 0;
        for (int iter = 0; iter < ITERS; ++iter) {
            auto t0 = high_resolution_clock::now();
            sve_lu();
            auto t1 = high_resolution_clock::now();
            t_sve_lu += duration_cast<nanoseconds>(t1 - t0).count() / 1000.0;
        }
        t_sve_lu /= ITERS;

        // ====== SVE 归约阶段 ======
        auto sve_reduce = [&]() {
            svfloat32_t acc = svdup_n_f32(0.0f);
            int k = 0;
            svbool_t pg = svwhilelt_b32(k, L);
            while (svptest_any(svptrue_b32(), pg)) {
                acc = svadd_f32_z(pg, acc,
                    svld1_gather_u32index_f32(pg, table.data(),
                        svld1_u32(pg, &sve_idx_buf[k])));
                k += svcntw();
                pg = svwhilelt_b32(k, L);
            }
            volatile float sink = svaddv_f32(svptrue_b32(), acc);
            (void)sink;
        };
        // 归约时间 = 全流程（含查表+归约） - 查表累加时间
        // 单独测 svaddv_f32 意义不大，以 total-(idx+lu) 推算

        double t_sve_reduce = t_sve_total - t_sve_idx - t_sve_lu;
        // 防止负值噪声
        if (t_sve_reduce < 0) t_sve_reduce = 0;

        double speedup = t_scalar_total / t_sve_total;

        std::cout << std::left
                  << std::setw(10) << L
                  << std::setw(14) << std::fixed << std::setprecision(3) << t_scalar_total
                  << std::setw(14) << std::fixed << std::setprecision(3) << t_scalar_idx
                  << std::setw(14) << std::fixed << std::setprecision(3) << t_scalar_lu
                  << std::setw(14) << std::fixed << std::setprecision(3) << t_sve_total
                  << std::setw(14) << std::fixed << std::setprecision(3) << t_sve_idx
                  << std::setw(14) << std::fixed << std::setprecision(3) << t_sve_lu
                  << std::setw(14) << std::fixed << std::setprecision(3) << t_sve_reduce
                  << std::setw(12) << std::fixed << std::setprecision(2) << speedup << "×"
                  << "\n";
    }
    std::cout << "\n";
}


// ====================== Main ======================

int main(int argc, char *argv[]) {
    // 解析命令行：./fp8_server [N] 只运行实验 N
    int only_exp = 0;
    if (argc >= 2) only_exp = atoi(argv[1]);

    std::cout << "\n" << std::string(70, '#') << "\n";
    std::cout << "  FP8 查表法矩阵乘法 — 服务器性能测试\n";
    std::cout << "  平台: HiSilicon Kunpeng, SVE\n";
    std::cout << "  SVE vector width: svcntw()=" << svcntw() << ", svcntb()=" << svcntb() << "\n";
    std::cout << "  Cache: L1d=" << L1D_KiB << "KiB  L2=" << L2_KiB << "KiB  L3=70MiB/Node\n";
    std::cout << "  LUT: 256×256 float = 256 KiB (> L1d, 永远在 L2+)\n";
    print_numa_info();
    std::cout << std::string(70, '#') << "\n";

    srand(42);

    // ========= 实验 6/7 矩阵参数 =========
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

    // 实验6: L1 分组建表查表试验 (80 核, SVE)
    if (!only_exp || only_exp == 6)
        exp6_l1_grouped_lut(LUT.data(), A.data(), B_T.data(), C_ref.data(),
                            N1, S1, L1, 80);

    // 实验7: 单核 L1 小表微基准 (SVE 归约)
    if (!only_exp || only_exp == 7)
        exp7_single_core_l1_lookup();

    // 实验8: 两级查表优化 (float 子表 + 行并行)
    if (!only_exp || only_exp == 8)
        exp8_two_level_lookup(LUT.data(), A.data(), B_T.data(), C_ref.data(),
                               N1, S1, L1, 80);

    // 实验9: 矩阵加载对查表微基准的影响
    if (!only_exp || only_exp == 9)
        exp9_matrix_load_microbench();

    // 实验10: 纯查表微基准（预计算索引）
    // 默认 n_lookups=0 表示按实验7规则自动计算
    // 也可由命令行参数指定：./fp8_server 10 <n_lookups>
    if (!only_exp || only_exp == 10) {
        int n_lookups = 0;
        if (argc >= 3) n_lookups = atoi(argv[2]);
        exp10_pure_lookup_microbench(n_lookups);
    }

    // 实验11: SVE gather 查表微基准（预计算索引）
    // 用法：./fp8_server 11 [n_lookups]
    if (!only_exp || only_exp == 11) {
        int n_lookups = 0;
        if (argc >= 3) n_lookups = atoi(argv[2]);
        exp11_sve_gather_microbench(n_lookups);
    }

    // 实验12: 索引计算微基准 (uint8→uint32, (a<<8)+b)
    if (!only_exp || only_exp == 12)
        exp12_lookup_phase_bench();

    std::cout << "\n测试完成!\n";
    return 0;
}
