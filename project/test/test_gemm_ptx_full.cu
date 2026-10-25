// -----------------------------------------------------------------------------
// test_gemm_ptx_full.cu
// 全尺寸 (1024 x 4096 x 4096) 正确性测试:
//   校验 src/transformer_layer.cu 中 gemm_ptx (GEMM_MYSELF) 的 GEMM 结果。
//   C = A * B, A[MxK] half 行主序, B[KxN] half 行主序, C[MxN] float 行主序。
// CPU 参考采用 OpenMP 并行 + double 累加 (i-k-j 顺序, B 连续访问)。
// -----------------------------------------------------------------------------
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../include/transformer_layer.cuh"

// 确定性随机数发生器
static uint32_t g_seed = 20260914u;
static float next_float() {
    g_seed = g_seed * 1664525u + 1013904223u;
    uint32_t v = (g_seed >> 16) & 0xFFFF;
    return (static_cast<float>(v) / 32767.5f) - 1.0f;  // [-1, 1]
}

int main() {
    const int M = 1024, N = 4096, K = 4096;
    const size_t ea = static_cast<size_t>(M) * K;
    const size_t eb = static_cast<size_t>(K) * N;
    const size_t ec = static_cast<size_t>(M) * N;

    printf("=== gemm_ptx 全尺寸正确性测试: M=%d N=%d K=%d ===\n", M, N, K);
    printf("A: %.1f MB (half)  B: %.1f MB (half)  C: %.1f MB (float)\n",
           ea * sizeof(half) / 1e6, eb * sizeof(half) / 1e6, ec * sizeof(float) / 1e6);

    // 1. 生成数据
    std::vector<half>  hA(ea);
    std::vector<half>  hB(eb);
    std::vector<float> hC(ec);
    std::vector<double> hRef(ec);

    for (auto& x : hA) x = __float2half(next_float());
    for (auto& x : hB) x = __float2half(next_float());

    // 2. 调用被测核函数
    printf("[1/3] Launching GEMM_MYSELF (gemm_ptx) ...\n");
    GEMM_MYSELF(M, N, K, hA.data(), hB.data(), hC.data());
    printf("      GPU kernel done.\n");

    // 3. CPU 参考 (OpenMP 并行 + double 累加)
    printf("[2/3] Computing CPU reference (double, OpenMP) ...\n");
#ifdef _OPENMP
    printf("      OpenMP threads = %d\n", omp_get_max_threads());
#endif
    #pragma omp parallel
    {
        std::vector<double> acc(N);
        #pragma omp for schedule(static)
        for (int m = 0; m < M; ++m) {
            std::fill(acc.begin(), acc.end(), 0.0);
            for (int k = 0; k < K; ++k) {
                const double a = static_cast<double>(__half2float(hA[static_cast<size_t>(m) * K + k]));
                const half* Bk = &hB[static_cast<size_t>(k) * N];
                for (int n = 0; n < N; ++n) {
                    acc[n] += a * static_cast<double>(__half2float(Bk[n]));
                }
            }
            for (int n = 0; n < N; ++n) {
                hRef[static_cast<size_t>(m) * N + n] = acc[n];
            }
        }
    }
    printf("      CPU reference done.\n");

    // 4. 全量逐元素比较
    printf("[3/3] Comparing all %zu elements ...\n", ec);
    double max_abs = 0.0, max_rel = 0.0;
    size_t max_idx = 0;
    size_t num_nan = 0, num_over = 0;
    double ref_min = 1e30, ref_max = -1e30;

    for (size_t i = 0; i < ec; ++i) {
        double got = static_cast<double>(hC[i]);
        double ref = hRef[i];
        if (std::isnan(got) || std::isinf(got)) { ++num_nan; continue; }
        ref_min = std::min(ref_min, ref);
        ref_max = std::max(ref_max, ref);
        double abs_err = std::fabs(got - ref);
        double rel_err = abs_err / std::fmax(1.0, std::fabs(ref));
        if (abs_err > max_abs) { max_abs = abs_err; max_rel = rel_err; max_idx = i; }
        if (rel_err > 1e-3 && abs_err > 1e-2) ++num_over;
    }

    size_t max_m = max_idx / N;
    size_t max_n = max_idx % N;
    printf("--------------------------------------------\n");
    printf("结果统计:\n");
    printf("  ref 值域     : [%.4f, %.4f]\n", ref_min, ref_max);
    printf("  NaN/Inf 数量 : %zu\n", num_nan);
    printf("  最大绝对误差 : %.6e\n", max_abs);
    printf("  最大相对误差 : %.6e\n", max_rel);
    printf("  最大误差位置 : (%zu, %zu)  ref=%.6f  got=%.6f\n",
           max_m, max_n, hRef[max_idx], static_cast<double>(hC[max_idx]));
    printf("  超出容差元素 : %zu / %zu\n", num_over, ec);

    // 抽样展示几个点
    size_t samples[][2] = {{0, 0}, {0, 4095}, {1023, 0}, {1023, 4095},
                           {512, 2048}, {17, 123}};
    for (auto& s : samples) {
        size_t idx = s[0] * N + s[1];
        printf("  C[%4zu][%4zu] = %.6f  (ref %.6f)\n",
               s[0], s[1], static_cast<double>(hC[idx]), hRef[idx]);
    }

    const double REL_TOL = 1e-3, ABS_TOL = 1e-2;
    bool pass = (num_nan == 0) && (max_rel <= REL_TOL || max_abs <= ABS_TOL);
    printf("--------------------------------------------\n");
    printf("%s\n", pass ? "FULL-SIZE TEST PASSED" : "FULL-SIZE TEST FAILED");
    printf("============================================\n");
    return pass ? 0 : 1;
}
