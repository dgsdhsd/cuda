#include "../include/memory_pool.cuh"
#include <iostream>

MemoryPool::MemoryPool(size_t max_size) : total_capacity(max_size), used_bytes(0) {
    // 1. 获取当前设备的默认内存池
    int device;
    cudaGetDevice(&device);
    cudaDeviceGetDefaultMemPool(&mem_pool, device);

    // 2. 设定显存池的释放阈值（防止它无脑囤积显存，这里设为 max_size）
    uint64_t threshold = max_size;
    cudaMemPoolSetAttribute(mem_pool, cudaMemPoolAttrReleaseThreshold, &threshold);
}

MemoryPool::~MemoryPool() {
    // 原生内存池会在设备重置时自动清理，不需要手动遍历
}

float* MemoryPool::allocate(size_t elements, cudaStream_t stream) {
    size_t bytes = elements * sizeof(float);
    
    // 你的自定义 OOM 拦截器 (用于触发 Checkpointing 重算逻辑)
    if (used_bytes + bytes > total_capacity) {
        throw std::runtime_error("OOM: Simulated Memory Pool Exhausted!");
    }
    
    float* d_ptr = nullptr;
    // 使用异步显存池分配，不阻塞 CPU，极大提升算术强度
    cudaError_t err = cudaMallocAsync((void**)&d_ptr, bytes, stream);
    
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA Native OOM: " + std::string(cudaGetErrorString(err)));
    }
    
    used_bytes += bytes;
    return d_ptr;
}

void MemoryPool::deallocate(float* ptr, size_t elements, cudaStream_t stream) {
    if (ptr) {
        // 异步归还给显存池，瞬间可用
        cudaFreeAsync(ptr, stream);
        used_bytes -= (elements * sizeof(float));
    }
}

size_t MemoryPool::get_free_space() const {
    return total_capacity - used_bytes;
}