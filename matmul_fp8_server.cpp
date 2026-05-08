// ====================== 实验11：SVE gather+累加联合微基准 ======================

/**
 * 实验11：匹配 lookup_sve 的查表实现方式。
 *
 * 与实验10的区别：
 *   实验10: 标量查表到 local_vals → SVE 累加 → 归约（3 阶段）
 *   实验11: SVE gather+累加 联合循环 → 归约（2 阶段，匹配 lookup_sve）
 *
 * 关键差异：
 *   - 无 local_vals 中间缓冲区，gather 结果直接累加到 acc
 *   - 使用 svadd_f32_z（zeroing 谓词，匹配 lookup_sve）
 */
void exp11_sve_gather_microbench(int n_lookups_user = 0) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "实验11: SVE gather+累加联合微基准\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "匹配 lookup_sve: svld1_gather + svadd_f32_z 联合循环, 无中间缓冲区\n";
    std::cout << "与实验10区别: 标量查表→SVE累加(3阶段) → SVE gather+累加联合(2阶段)\n\n";

    constexpr int TABLE_B = 256;
    constexpr int MAX_LOOKUPS = 128;
    constexpr int TABLE_A_VALS[] = {2, 4, 8, 16, 32};
    constexpr int WARMUP = 2000;
    constexpr int ITERS = 20000;

    std::cout << std::left
              << std::setw(16) << "表维度"
              << std::setw(14) << "表大小(KiB)"
              << std::setw(14) << "查表次数"
              << std::setw(16) << "平均耗时(μs)"
              << std::setw(14) << "总查表(ns)"
              << std::setw(14) << "每次查表(ns)"
              << std::setw(14) << "L1-miss%"
              << std::setw(14) << "gather+累加(μs)"
              << std::setw(14) << "归约(μs)"
              << std::setw(16) << "vs实验10"
              << "\n" << std::string(140, '-') << "\n";

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

        volatile float sink = 0;

        // Warmup
        for (int w = 0; w < WARMUP; ++w) {
            svfloat32_t acc = svdup_n_f32(0.0f);
            int t = 0;
            svbool_t pg = svwhilelt_b32(t, n_lookups);
            while (svptest_any(svptrue_b32(), pg)) {
                svuint32_t idx_vec = svld1_u32(pg, indices.data() + t);
                acc = svadd_f32_z(pg, acc,
                    svld1_gather_u32index_f32(pg, sub.data(), idx_vec));
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

        // 两阶段计时：gather+累加联合 → 归约
        double sum_total = 0, sum_ga = 0, sum_reduce = 0;
        for (int iter = 0; iter < ITERS; ++iter) {
            auto t0 = high_resolution_clock::now();

            // 阶段1: SVE gather + 累加联合循环（匹配 lookup_sve）
            svfloat32_t acc = svdup_n_f32(0.0f);
            int t = 0;
            svbool_t pg = svwhilelt_b32(t, n_lookups);
            while (svptest_any(svptrue_b32(), pg)) {
                svuint32_t idx_vec = svld1_u32(pg, indices.data() + t);
                acc = svadd_f32_z(pg, acc,
                    svld1_gather_u32index_f32(pg, sub.data(), idx_vec));
                t += svcntw();
                pg = svwhilelt_b32(t, n_lookups);
            }
            auto t1 = high_resolution_clock::now();

            // 阶段2: SVE 归约
            sink = svaddv_f32(svptrue_b32(), acc);
            auto t2 = high_resolution_clock::now();

            sum_ga     += duration_cast<nanoseconds>(t1 - t0).count() / 1000.0;
            sum_reduce += duration_cast<nanoseconds>(t2 - t1).count() / 1000.0;
            sum_total  += duration_cast<nanoseconds>(t2 - t0).count() / 1000.0;
        }

        if (perf_ok) {
            pc_l1_acc.disable();
            pc_l1_miss.disable();
        }

        double avg_us = sum_total / ITERS;
        double avg_ns_total = avg_us * 1000;
        double avg_ns_each = avg_ns_total / n_lookups;
        double avg_ga_us = sum_ga / ITERS;
        double avg_reduce_us = sum_reduce / ITERS;

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
                  << std::setw(14) << std::fixed << std::setprecision(4) << avg_ga_us
                  << std::setw(14) << std::fixed << std::setprecision(4) << avg_reduce_us
                  << std::setw(16) << "—"
                  << "\n";
    }
    std::cout << std::string(140, '-') << "\n";
    std::cout << "注: gather+累加联合 = svld1_gather_u32index_f32 + svadd_f32_z（匹配 lookup_sve）\n";
    std::cout << "    无 local_vals 中间缓冲区，gather 结果直接写入 acc 向量寄存器\n\n";
}
