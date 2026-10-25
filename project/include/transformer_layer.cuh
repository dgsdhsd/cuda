#pragma once
#include "memory_pool.cuh"
#include "stream_manager.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda.h>

class TransformerLayerNode {
public:
    const size_t M = 1024;
    const size_t K = 4096;
    const size_t N = 4096;

    // 设备端激活值
    float* d_input = nullptr;   // 输入 A [M, K]
    float* d_output = nullptr;  // 输出 C [M, N]
    
    // 1. 采用 Fused QKV 物理布局：显存连续包含 Wq, Wk, Wv (总大小: K * 3N)
    float* d_Wqkv = nullptr;    
    
    // 2. 逻辑视图指针：给不喜欢 Fused 接口的 Kernel 预留逻辑切片（零额外开销）
    float* d_Wq = nullptr; 
    float* d_Wk = nullptr;
    float* d_Wv = nullptr;

    void* qlora_weight_B = nullptr; // QLoRA 预留

    bool use_checkpointing;
    bool is_activation_saved;

    explicit TransformerLayerNode(bool enable_checkpoint);
    ~TransformerLayerNode();

    /**
     * @brief 极致简化的初始化函数
     * @param h_Wqkv_packed Host端已经拼接好的 QKV 连续数据 (大小: K * 3N * sizeof(float))
     *                     必须是 Pinned Memory 才能发挥最高 PCIe 吞吐
     * @param mem_pool 显存池
     * @param stream 传入单个主初始化流即可
     */
    void initialize_weights_unified(
        const float* h_Wqkv_packed,
        MemoryPool& mem_pool,
        cudaStream_t stream
    );

    void forward(MemoryPool& mem_pool, cudaStream_t stream);
    void backward(MemoryPool& mem_pool, cudaStream_t stream);
};

// -----------------------------------------------------------------------------
// 自研 TMA + ldmatrix + mma.sync 的 FP16 Tensor-Core GEMM 对外接口
// (实现位于 src/transformer_layer.cu)
//   输入: hostA[M*K] 行主序 half, hostB[K*N] 行主序 half
//   输出: hostC[M*N] 行主序 float, 满足 C = A * B
//   约束: M % 128 == 0, N % 64 == 0, K % 64 == 0
// -----------------------------------------------------------------------------
void GEMM_MYSELF(int M, int N, int K, const half* hostA, const half* hostB, float* hostC);
constexpr int TILE_M    = 128; // 每 block 处理 128 行
constexpr int TILE_N    = 64;  // 每 block 处理 64 列
constexpr int TILE_K    = 64;  // 每 stage 沿 K 维度取 64 个 half (128B)
constexpr int NUM_STAGE = 4;   // 双缓冲 Stage 数量

constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
constexpr int N8    = TILE_N / MMA_N;            // 64 / 8  = 8 个 MMA 列块
constexpr int KS    = TILE_K / MMA_K;            // 64 / 16 = 4 个 MMA K 块
constexpr int WARPS = TILE_M / MMA_M;            // 128 / 16 = 8 warps (256 线程)

// -----------------------------------------------------------------------------
// PTX 汇编与辅助函数
// -----------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint32_t m, uint32_t cnt) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(m), "r"(cnt));
}

__device__ __forceinline__ void mbar_ex(uint32_t m, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" :: "r"(m), "r"(bytes));
}

__device__ __forceinline__ void mbar_wait(uint32_t m, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "WAIT_LOOP%=:\n"
        "mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
        "@!p bra WAIT_LOOP%=;\n"
        "}\n"
        :: "r"(m), "r"(phase)
    );
}

__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* map, int i0, int i1, uint32_t mb) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(dst), "l"(map), "r"(i0), "r"(i1), "r"(mb)
        : "memory"
    );
}

__device__ __forceinline__ void ldmatrix_a_x4(unsigned& a0, unsigned& a1, unsigned& a2, unsigned& a3, uint32_t smem) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
        : "r"(smem)
    );
}

__device__ __forceinline__ void ldmatrix_b_x2_trans(unsigned& b0, unsigned& b1, uint32_t smem) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(b0), "=r"(b1)
        : "r"(smem)
    );
}

__device__ __forceinline__ void mma_sync_m16n8k16(
    float& c0, float& c1, float& c2, float& c3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
}

__device__ __forceinline__ uint32_t swizzle128(uint32_t logical_byte) {
    uint32_t row = logical_byte / 128;
    uint32_t off = logical_byte % 128;
    return row * 128 + (off ^ ((row % 8) * 16));
}
// -----------------------------------------------------------------------------
// 补充 PTX 汇编接口：mbarrier.arrive
// -----------------------------------------------------------------------------
__device__ __forceinline__ void mbar_arrive(uint32_t m) {
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" :: "r"(m));
}


#define CUDA_CHECK(c) do { \
    cudaError_t _e = (c); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "%s:%d CUDA err: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

#define CUK(c) do { \
    CUresult _r = (c); \
    if (_r != CUDA_SUCCESS) { \
        const char* _s; \
        cuGetErrorString(_r, &_s); \
        fprintf(stderr, "%s:%d Driver err %d: %s\n", __FILE__, __LINE__, (int)_r, _s); \
        exit(1); \
    } \
} while(0)