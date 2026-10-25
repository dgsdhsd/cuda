#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <string>

class MemoryPool {
private:
    size_t total_capacity;
    size_t used_bytes; // 我们依然手动追踪使用量，用来故意触发 OOM 测试
    cudaMemPool_t mem_pool; // CUDA 原生内存池句柄

public:
    explicit MemoryPool(size_t max_size);
    ~MemoryPool();

    // 必须传入 stream，实现与计算完全重叠的纳秒级分配
    float* allocate(size_t elements, cudaStream_t stream);
    void deallocate(float* ptr, size_t elements, cudaStream_t stream);
    
    size_t get_free_space() const;
};