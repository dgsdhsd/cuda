#include <iostream>
#include <vector>

// 直接引入你工程里的公共头文件
#include "common.h"

int main() {
    const int M = 4, K = 3, N = 5;

    std::vector<float> h_A = {1.0f, 4.0f, 7.0f, 10.0f, 2.0f, 5.0f, 8.0f, 11.0f, 3.0f, 6.0f, 9.0f, 12.0f};
    std::vector<float> h_B = {1.0f, 0.0f, 1.0f, 0.0f, 2.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 2.0f, 1.0f, 0.0f};
    std::vector<float> h_C(M * N, 0.0f);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    // 干净利落的错误检查
    CHECK_CUDA(cudaMalloc((void**)&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_C, M * N * sizeof(float)));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    CHECK_CUBLAS(cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, 
        M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M
    ));

    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Calculated successfully!\n";

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}