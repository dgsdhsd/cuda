#include <iostream>
#include "memory_pool.cuh"
#include "transformer_layer.cuh"
#include "stream_manager.cuh"
#include "common.h"

int main() {
    std::cout << "=== CUDA Transformer Layer Test Pass ===" << std::endl;

    // 1. 初始化显存池 (200 MB) 与 流管理器 (4 个流)
    MemoryPool mem_pool(1000 * 1024 * 1024);
    StreamManager stream_manager(4);

    // 2. 创建 Transformer 层 (开启 Checkpointing)
    TransformerLayerNode layer(true);

    // 3. 申请单块打包的 Pinned Memory
    size_t single_elements = layer.K * layer.N;
    size_t total_bytes = 3 * single_elements * sizeof(float);

    float *h_Wqkv_packed = nullptr;
    cudaMallocHost(&h_Wqkv_packed, total_bytes);

    // 4. 填充固定初值（方便后续打印矩阵或与 CPU 结果对照验算）
    // Wq 全 1.0f | Wk 全 2.0f | Wv 全 3.0f
    for (size_t i = 0; i < single_elements; ++i) {
        h_Wqkv_packed[i]                       = 1.0f; // Wq
        h_Wqkv_packed[single_elements + i]     = 2.0f; // Wk
        h_Wqkv_packed[2 * single_elements + i] = 3.0f; // Wv
    }

    // 5. 权重传输 (在 Stream 0 上发起 H2D)
    std::cout << "[Init] Direct H2D weight initialization..." << std::endl;
    layer.initialize_weights_unified(h_Wqkv_packed, mem_pool, stream_manager.get_stream(0));

    // 6. 前向计算 (在 Stream 3 上发射 GEMM Kernel)
    std::cout << "[Forward] Executing GEMM Forward pass..." << std::endl;
    layer.forward(mem_pool, stream_manager.get_stream(3));

    // 7. 反向传播
    std::cout << "[Backward] Executing Backward pass..." << std::endl;
    layer.backward(mem_pool, stream_manager.get_stream(3));

    // 8. 同步等待完成
    stream_manager.synchronize_all();
    std::cout << "[Done] All operations completed successfully!" << std::endl;

    // 9. 释放 Host 内存
    cudaFreeHost(h_Wqkv_packed);

    return 0;
}