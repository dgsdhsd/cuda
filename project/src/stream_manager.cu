#include "stream_manager.cuh"
#include <stdexcept>

StreamManager::StreamManager(size_t stream_count) {
    streams.resize(stream_count);
    for (size_t i = 0; i < stream_count; ++i) {
        cudaStreamCreate(&streams[i]);
    }
}

StreamManager::~StreamManager() {
    for (auto stream : streams) {
        cudaStreamDestroy(stream);
    }
}

cudaStream_t StreamManager::get_stream(size_t index) const {
    return streams[index];
}

void StreamManager::synchronize_all() const {
    for (auto stream : streams) {
        cudaStreamSynchronize(stream);
    }
}