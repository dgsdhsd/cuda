#pragma once
#include <cuda_runtime.h>
#include <vector>

class StreamManager {
private:
    std::vector<cudaStream_t> streams;

public:
    explicit StreamManager(size_t stream_count);
    ~StreamManager();

    // 获取指定的 CUDA 流
    cudaStream_t get_stream(size_t index) const;

    // 全局同步
    void synchronize_all() const;
};
