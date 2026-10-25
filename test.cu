#include <iostream>
#include <cuda_runtime.h>

// CUDA 核函数：在 GPU 上运行
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int numElements = 1024;
    size_t size = numElements * sizeof(float);

    // 1. 在主机（CPU）上分配内存
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // 初始化数据
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // 2. 在设备（GPU）上分配显存
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // 3. 将数据从 CPU 内存拷贝到 GPU 显存
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 4. 启动 GPU 线程跑核函数
    // 启动 1 个线程块（Block），每个块包含 1024 个线程（Thread）
    int threadsPerBlock = 1024;
    int blocksPerGrid = 1;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    // 5. 将计算结果从 GPU 拷贝回 CPU
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 6. 验证结果是否正确（应该全是 1.0 + 2.0 = 3.0）
    bool success = true;
    for (int i = 0; i < numElements; ++i) {
        if (h_C[i] != 3.0f) {
            success = false;
            break;
        }
    }

    // 打印测试结果
    std::cout << "========================================" << std::endl;
    if (success) {
        std::cout << "🎉 CUDA 测试成功! GPU 矢量计算结果完全正确！" << std::endl;
    } else {
        std::cout << "❌ 测试失败，计算数值不匹配。" << std::endl;
    }
    std::cout << "========================================" << std::endl;

    // 释放内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
