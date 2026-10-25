// -----------------------------------------------------------------------------
// test_gemm_ptx.cu
// 独立正确性测试：校验 project/src/transformer_layer.cu 中 gemm_ptx 核函数
// (通过 GEMM_MYSELF 封装接口) 的 GEMM 计算结果是否与 CPU 参考一致。
//
// 计算语义: C = A * B
//   A: M x K, half, 行主序
//   B: K x N, half, 行主序
//   C: M x N, float, 行主序
// 约束: M % 128 == 0, N % 64 == 0, K % 64 == 0
// -----------------------------------------------------------------------------
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "../include/transformer_layer.cuh"

// 简单确定性随机数发生器，避免不同平台 rand() 差异
static uint32_t g_seed = 20260914u;
static float next_float() {
    g_seed = g_seed * 1664525u + 1013904223u;
    // 取高 16 位映射到 [-1, 1]
    uint32_t v = (g_seed >> 16) & 0xFFFF;
    return (static_cast<float>(v) / 32767.5f) - 1.0f;
}

// CPU 参考实现 (double 累加，作为"真值"基准)
static void gemm_ref_cpu(int M, int N, int K,
                         const half* A, const half* B, double* C) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                acc += static_cast<double>(__half2float(A[static_cast<size_t>(m) * K + k])) *
                       static_cast<double>(__half2float(B[static_cast<size_t>(k) * N + n]));
            }
            C[static_cast<size_t>(m) * N + n] = acc;
        }
    }
}

static bool run_case(int M, int N, int K) {
    std::vector<half>  hA(static_cast<size_t>(M) * K);
    std::vector<half>  hB(static_cast<size_t>(K) * N);
    std::vector<float> hC(static_cast<size_t>(M) * N);
    std::vector<double> hRef(static_cast<size_t>(M) * N);

    g_seed = 20260914u; // 每个 case 使用相同随机序列，便于复现
    for (auto& x : hA) x = __float2half(next_float());
    for (auto& x : hB) x = __float2half(next_float());

    // 调用被测核函数 (定义于 src/transformer_layer.cu)
    GEMM_MYSELF(M, N, K, hA.data(), hB.data(), hC.data());

    // CPU 参考
    gemm_ref_cpu(M, N, K, hA.data(), hB.data(), hRef.data());

    // 逐元素比较
    double max_abs = 0.0, max_rel = 0.0;
    int    max_i = -1, max_j = -1;
    bool   has_nan = false;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double got = static_cast<double>(hC[static_cast<size_t>(m) * N + n]);
            double ref = hRef[static_cast<size_t>(m) * N + n];
            if (std::isnan(got) || std::isinf(got)) has_nan = true;
            double abs_err = std::fabs(got - ref);
            double rel_err = abs_err / std::fmax(1.0, std::fabs(ref));
            if (abs_err > max_abs) { max_abs = abs_err; max_rel = rel_err; max_i = m; max_j = n; }
            else if (abs_err == max_abs && rel_err > max_rel) { max_rel = rel_err; max_i = m; max_j = n; }
        }
    }

    // 判定阈值: FP16 输入 + FP32 累加, 相对误差应远小于 1e-3
    const double REL_TOL = 1e-3;
    const double ABS_TOL = 1e-3;
    bool pass = !has_nan && (max_rel <= REL_TOL || max_abs <= ABS_TOL);

    printf("[%s] M=%d N=%d K=%d | max_abs=%.3e max_rel=%.3e%s | @(%d,%d) ref=%.4f got=%.4f\n",
           pass ? "PASS" : "FAIL", M, N, K,
           max_abs, max_rel, has_nan ? " [NaN/Inf!]" : "",
           max_i, max_j,
           (max_i >= 0) ? hRef[static_cast<size_t>(max_i) * N + max_j] : 0.0,
           (max_i >= 0) ? hC[static_cast<size_t>(max_i) * N + max_j] : 0.0);
    return pass;
}

int main() {
    printf("=== gemm_ptx 正确性测试 (C = A*B, FP16 in / FP32 out) ===\n");

    struct Case { int M, N, K; };
    const std::vector<Case> cases = {
        {128,   64,   64 },   // 单 block, 单 stage
        {256,  128,  256 },   // 多 block, 4 stage 满流水
        {512,  256,  512 },   // 更大尺寸
        {1024, 512,  1024},   // 多 block + 多 stage
        {512,  512,  2048},   // 长 K 累加
    };

    bool all_pass = true;
    for (auto& c : cases) {
        all_pass = run_case(c.M, c.N, c.K) && all_pass;
    }

    printf("============================================\n");
    printf("%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    printf("============================================\n");
    return all_pass ? 0 : 1;
}
