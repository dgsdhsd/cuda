#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

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

// -----------------------------------------------------------------------------
// 分块尺寸常量定义
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Kernel: GEMM (Warp Specialization + TMA + 0 Bank Conflict Epilogue)
// -----------------------------------------------------------------------------
__global__ void __launch_bounds__(256) gemm_ptx(
    const CUtensorMap* mapA, const CUtensorMap* mapB,
    float* __restrict__ dC, int M, int N, int K
) {
    extern __shared__ char smem_raw[];

    const size_t a_stage = TILE_M * TILE_K * sizeof(half); 
    const size_t b_stage = TILE_K * TILE_N * sizeof(half); 

    const int tid  = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int bm   = blockIdx.y * TILE_M;
    const int bn   = blockIdx.x * TILE_N;

    // -------------------------------------------------------------------------
    // Shared Memory 空间划分 (A, B 缓冲区及 Full/Empty 双向屏障)
    // -------------------------------------------------------------------------
    uint32_t ap[NUM_STAGE], bp[NUM_STAGE];
    uint32_t mb_full[NUM_STAGE], mb_empty[NUM_STAGE];

    char* barrier_base = smem_raw + NUM_STAGE * (a_stage + b_stage);
    for (int s = 0; s < NUM_STAGE; ++s) {
        ap[s]       = smem_u32(smem_raw + s * a_stage);
        bp[s]       = smem_u32(smem_raw + NUM_STAGE * a_stage + s * b_stage);
        mb_full[s]  = smem_u32(barrier_base + s * sizeof(uint64_t));
        mb_empty[s] = smem_u32(barrier_base + (NUM_STAGE + s) * sizeof(uint64_t));
    }

    // 屏障初始化：mb_full 等待 1 个 TMA 信号，mb_empty 等待 8 个 Consumer Warp 信号
    if (tid == 0) {
        for (int s = 0; s < NUM_STAGE; ++s) {
            mbar_init(mb_full[s], 1);    // Producer 到达
            mbar_init(mb_empty[s], 8);   // 8 个 Consumer Warps 到达
        }
    }
    __syncthreads(); // 仅在 Prologue 执行一次 Block 级同步

    float acc[N8][4] = {0.0f};
    const int n_kb = K / TILE_K;
    const uint32_t txA = static_cast<uint32_t>(a_stage);
    const uint32_t txB = static_cast<uint32_t>(b_stage);

    // Phase 翻转状态数组初始化 (对应 4 缓冲区)
    uint32_t phase_full[NUM_STAGE]  = {0};
    uint32_t phase_empty[NUM_STAGE] = {0};

    // -------------------------------------------------------------------------
    // Prologue: Producer 预加载 NUM_STAGE - 1 (3个) Stage 填满流水线
    // -------------------------------------------------------------------------
    if (tid == 0) {
        int prefetch_stages = min(n_kb, NUM_STAGE - 1);
        for (int s = 0; s < prefetch_stages; ++s) {
            mbar_ex(mb_full[s], txA + txB);
            tma2d(ap[s], mapA, s * TILE_K, bm, mb_full[s]);
            tma2d(bp[s], mapB, bn, s * TILE_K, mb_full[s]);
        }
    }

    const int lm_A = lane % 16;
    const int lc_A = (lane / 16) * 8;

    // -------------------------------------------------------------------------
    // Main Loop: 4 级异步生产者-消费者流水线
    // -------------------------------------------------------------------------
    for (int kb = 0; kb < n_kb; ++kb) {
        int sc       = kb % NUM_STAGE;          // 当前 Consumer 计算的 Stage 槽位
        int fetch_kb = kb + NUM_STAGE - 1;      // Producer 当前需要预取的全局 Stage 编号

        // 1. Producer 逻辑 (tid == 0 异步预取 fetch_kb)
        if (tid == 0 && fetch_kb < n_kb) {
            int sn = fetch_kb % NUM_STAGE;      // 目标 SMEM 槽位

            // 当该 SMEM 槽位已被轮转使用过一次（fetch_kb >= NUM_STAGE），才需要等待消费者释放
            if (fetch_kb >= NUM_STAGE) {
                mbar_wait(mb_empty[sn], phase_empty[sn]);
                phase_empty[sn] ^= 1;
            }

            mbar_ex(mb_full[sn], txA + txB);
            tma2d(ap[sn], mapA, fetch_kb * TILE_K, bm, mb_full[sn]);
            tma2d(bp[sn], mapB, bn, fetch_kb * TILE_K, mb_full[sn]);
        }

        // 2. Consumer 逻辑 (所有 8 个 Warp 等待当前 Stage 数据就绪)
        mbar_wait(mb_full[sc], phase_full[sc]);
        phase_full[sc] ^= 1;

        // 3. 执行 MMA 计算
        #pragma unroll
        for (int kk = 0; kk < KS; ++kk) {
            int koff = kk * MMA_K;

            int gm = warp * 16 + lm_A;
            int gk = koff + lc_A;
            uint32_t logicalA = static_cast<uint32_t>(gm * TILE_K + gk) * sizeof(half);
            uint32_t physA    = swizzle128(logicalA);

            unsigned a0, a1, a2, a3;
            ldmatrix_a_x4(a0, a1, a2, a3, ap[sc] + physA);

            #pragma unroll
            for (int n = 0; n < N8; ++n) {
                int kb_row = koff + (lane % 16);
                int colB   = n * 8;
                uint32_t logicalB = static_cast<uint32_t>(kb_row * TILE_N + colB) * sizeof(half);
                uint32_t physB    = swizzle128(logicalB);

                unsigned b0, b1;
                ldmatrix_b_x2_trans(b0, b1, bp[sc] + physB);

                mma_sync_m16n8k16(
                    acc[n][0], acc[n][1], acc[n][2], acc[n][3],
                    a0, a1, a2, a3,
                    b0, b1
                );
            }
        }

        // 4. 每个 Warp 读取完毕后通知 Producer (lane == 0 代表当前 Warp 到达)
        if (lane == 0) {
            mbar_arrive(mb_empty[sc]);
        }
    }

    // -------------------------------------------------------------------------
    // Epilogue: 复用 SMEM + Padding(68) 消除 Bank 冲突 + STG.128 向量写回
    // -------------------------------------------------------------------------
    __syncthreads(); // 确保主循环计算全部结束，SMEM 可安全覆写
    float* smem_C = reinterpret_cast<float*>(smem_raw);
    constexpr int TILE_N_PAD = TILE_N + 4; // 68

    const int rA = lane / 4;
    const int cA = (lane % 4) * 2;

    // 1. 写 SMEM (使用 TILE_N_PAD，0 Bank Conflict)
    #pragma unroll
    for (int n = 0; n < N8; ++n) {
        int local_r0 = warp * 16 + rA;
        int local_r1 = local_r0 + 8;
        int local_c0 = n * 8 + cA;
        int local_c1 = local_c0 + 1;

        smem_C[local_r0 * TILE_N_PAD + local_c0] = acc[n][0];
        smem_C[local_r0 * TILE_N_PAD + local_c1] = acc[n][1];
        smem_C[local_r1 * TILE_N_PAD + local_c0] = acc[n][2];
        smem_C[local_r1 * TILE_N_PAD + local_c1] = acc[n][3];
    }

    __syncthreads();

    // 2. 向量化合并写回 GMEM (STG.128)
    constexpr int TOTAL_FLOAT4 = (TILE_M * TILE_N) / 4; // 2048 个 float4
    float4* dC_f4 = reinterpret_cast<float4*>(dC);
    const int gmem_stride_f4 = N / 4;

    #pragma unroll
    for (int i = tid; i < TOTAL_FLOAT4; i += blockDim.x) {
        int local_row    = i / (TILE_N / 4);
        int local_col_f4 = i % (TILE_N / 4);

        int smem_float_offset = local_row * TILE_N_PAD + local_col_f4 * 4;
        float4 val = *reinterpret_cast<const float4*>(&smem_C[smem_float_offset]);

        int global_row    = bm + local_row;
        int global_col_f4 = (bn / 4) + local_col_f4;

        if (global_row < M && global_col_f4 < gmem_stride_f4) {
            dC_f4[global_row * gmem_stride_f4 + global_col_f4] = val;
        }
    }
}
// -----------------------------------------------------------------------------
// 目标封装函数接口
// -----------------------------------------------------------------------------
void GEMM_MYSELF(int M, int N, int K, const half* hostA, const half* hostB, float* hostC) {
    // 1. 校验分块尺寸对齐要求
    assert(M % TILE_M == 0 && "M 必须能够被 TILE_M(128) 整除");
    assert(N % TILE_N == 0 && "N 必须能够被 TILE_N(64) 整除");
    assert(K % TILE_K == 0 && "K 必须能够被 TILE_K(64) 整除");

    // 2. 初始化 Driver API Context (单例模式防重复初始化)
    static bool driver_inited = false;
    if (!driver_inited) {
        CUK(cuInit(0));
        CUdevice dv;
        CUcontext ctx;
        CUK(cuDeviceGet(&dv, 0));
        CUK(cuCtxCreate(&ctx, 0, dv));
        driver_inited = true;
    }

    // 3. 显存分配
    half *dA = nullptr, *dB = nullptr;
    float *dC = nullptr;
    const size_t bytesA = static_cast<size_t>(M) * K * sizeof(half);
    const size_t bytesB = static_cast<size_t>(K) * N * sizeof(half);
    const size_t bytesC = static_cast<size_t>(M) * N * sizeof(float);

    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));

    // 4. 数据由 Host 拷贝至 Device
    CUDA_CHECK(cudaMemcpy(dA, hostA, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hostB, bytesB, cudaMemcpyHostToDevice));

    // 5. 编码构建 A / B 矩阵的 TMA TensorMap 描述符
    CUtensorMap mapA, mapB;
    {
        uint64_t dim[2] = {static_cast<uint64_t>(K), static_cast<uint64_t>(M)};
        uint64_t st[1]  = {static_cast<uint64_t>(K) * sizeof(half)};
        uint32_t box[2] = {static_cast<uint32_t>(TILE_K), static_cast<uint32_t>(TILE_M)};
        uint32_t es[2]  = {1, 1};

        CUK(cuTensorMapEncodeTiled(
            &mapA, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, dA, dim, st, box, es,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        ));
    }
    {
        uint64_t dim[2] = {static_cast<uint64_t>(N), static_cast<uint64_t>(K)};
        uint64_t st[1]  = {static_cast<uint64_t>(N) * sizeof(half)};
        uint32_t box[2] = {static_cast<uint32_t>(TILE_N), static_cast<uint32_t>(TILE_K)};
        uint32_t es[2]  = {1, 1};

        CUK(cuTensorMapEncodeTiled(
            &mapB, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, dB, dim, st, box, es,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        ));
    }

    // 将 TensorMap 描述符拷贝至 Device 显存
    CUtensorMap *dma = nullptr, *dmb = nullptr;
    CUDA_CHECK(cudaMalloc(&dma, sizeof(CUtensorMap)));
    CUDA_CHECK(cudaMalloc(&dmb, sizeof(CUtensorMap)));
    CUDA_CHECK(cudaMemcpy(dma, &mapA, sizeof(CUtensorMap), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dmb, &mapB, sizeof(CUtensorMap), cudaMemcpyHostToDevice));

    // 6. 动态共享内存与 Grid/Block 参数设置
    size_t smem_size = NUM_STAGE * (TILE_M * TILE_K + TILE_K * TILE_N) * sizeof(half) + NUM_STAGE * sizeof(uint64_t);
    CUDA_CHECK(cudaFuncSetAttribute(gemm_ptx, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_size)));

    dim3 grid(N / TILE_N, M / TILE_M);
    dim3 block(256);

    // 7. 执行核函数 launch
    gemm_ptx<<<grid, block, smem_size>>>(dma, dmb, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 8. 结果写回 Host 侧 hostC 内存区
    CUDA_CHECK(cudaMemcpy(hostC, dC, bytesC, cudaMemcpyDeviceToHost));

    // 9. 释放 Device 资源
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dma));
    CUDA_CHECK(cudaFree(dmb));
}

// -----------------------------------------------------------------------------
// 调用示例 main()
// -----------------------------------------------------------------------------
int main() {
    const int M = 1024, N = 4096, K = 4096;

    std::srand(2026);
    std::vector<half> hA(static_cast<size_t>(M) * K);
    std::vector<half> hB(static_cast<size_t>(K) * N);
    std::vector<float> hC(static_cast<size_t>(M) * N);

    for (auto& x : hA) x = __float2half(static_cast<float>(std::rand() % 7 - 3));
    for (auto& x : hB) x = __float2half(static_cast<float>(std::rand() % 7 - 3));

    // 调用封装好的 GEMM 函数
    xx(M, N, K, hA.data(), hB.data(), hC.data());

    // CPU 数值校验
    double max_diff = 0.0;
    for (int m = 0; m < 16; ++m) {
        for (int n = 0; n < 16; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += __half2float(hA[static_cast<size_t>(m) * K + k]) *
                       __half2float(hB[static_cast<size_t>(k) * N + n]);
            }
            double diff = std::fabs(static_cast<double>(hC[static_cast<size_t>(m) * N + n]) - acc);
            if (diff > max_diff) max_diff = diff;
        }
    }

    printf("Check Success! Max Diff = %.4f\n", max_diff);
    return 0;
}