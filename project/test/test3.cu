// =============================================================================
// test3_fixed.cu —— 裸 mma.sync + ldmatrix + TMA(SWIZZLE_128B) + 生产-消费双缓冲
// sm_120 (Blackwell) fp16 GEMM, 目标 M=1024 x N=4096 x K=4096
// =============================================================================
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
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
constexpr int TILE_N    = 64;  // 每 block 处理 64 列 (SWIZZLE_128B 要求内层 <= 128B)
constexpr int TILE_K    = 64;  // 每 stage 沿 K 维度取 64 个 half (128B)
constexpr int NUM_STAGE = 2;   // 双缓冲 Stage 数量

constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16; // PTX mma.sync m16n8k16 规格
constexpr int N8    = TILE_N / MMA_N;            // 64 / 8  = 8 个 MMA 列块
constexpr int KS    = TILE_K / MMA_K;            // 64 / 16 = 4 个 MMA K 块
constexpr int WARPS = TILE_M / MMA_M;            // 128 / 16 = 8 warps (Block 256 线程)

// -----------------------------------------------------------------------------
// PTX 汇编与辅助函数
// -----------------------------------------------------------------------------

// 通用共享内存指针转 32 位 Shared 域地址
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// mbarrier 初始化与操作
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

// 2D TMA Async Copy
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* map, int i0, int i1, uint32_t mb) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(dst), "l"(map), "r"(i0), "r"(i1), "r"(mb)
        : "memory"
    );
}

// 矩阵 A 加载: ldmatrix.m8n8.x4 (不带 .trans) -> 保持 16x16 Row-Major 存入寄存器
__device__ __forceinline__ void ldmatrix_a_x4(unsigned& a0, unsigned& a1, unsigned& a2, unsigned& a3, uint32_t smem) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
        : "r"(smem)
    );
}

// 矩阵 B 加载: ldmatrix.m8n8.x2.trans (带 .trans) -> 从 Row-Major 转置为 Col-Major 存入寄存器
__device__ __forceinline__ void ldmatrix_b_x2_trans(unsigned& b0, unsigned& b1, uint32_t smem) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(b0), "=r"(b1)
        : "r"(smem)
    );
}

// PTX 裸 mma.sync.m16n8k16 指令 (row.col)
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

// SWIZZLE_128B 逻辑地址到 Shared Memory 物理地址转换
__device__ __forceinline__ uint32_t swizzle128(uint32_t logical_byte) {
    uint32_t row = logical_byte / 128;
    uint32_t off = logical_byte % 128;
    return row * 128 + (off ^ ((row % 8) * 16));
    //可以保证每16个字节连续。
}

// -----------------------------------------------------------------------------
// Kernel: GEMM 双缓冲 + TMA + ldmatrix + mma.sync
// -----------------------------------------------------------------------------
__global__ void __launch_bounds__(256) gemm_ptx(
    const CUtensorMap* mapA, const CUtensorMap* mapB,
    float* __restrict__ dC, int M, int N, int K
) {
    extern __shared__ char smem_raw[];

    // 共享内存 Block 内逻辑划分
    const size_t a_stage = TILE_M * TILE_K * sizeof(half); // 128 * 64 * 2 = 16384 B
    const size_t b_stage = TILE_K * TILE_N * sizeof(half); // 64 * 64 * 2  = 8192 B

    const int tid  = threadIdx.x;
    const int warp = tid / 32;
    const int bm   = blockIdx.y * TILE_M;
    const int bn   = blockIdx.x * TILE_N;

    // 按 Byte 正确计算 Stage 基地址 (避免类型指针缩放错误)
    uint32_t ap[NUM_STAGE], bp[NUM_STAGE], mb[NUM_STAGE];
    for (int s = 0; s < NUM_STAGE; ++s) {
        ap[s] = smem_u32(smem_raw + s * a_stage);
        bp[s] = smem_u32(smem_raw + NUM_STAGE * a_stage + s * b_stage);
        mb[s] = smem_u32(smem_raw + NUM_STAGE * (a_stage + b_stage) + s * sizeof(uint64_t));
    }

    // 初始化 mbarrier
    if (tid == 0) {
        for (int s = 0; s < NUM_STAGE; ++s) {
            mbar_init(mb[s], 1);
        }
    }
    __syncthreads();

    // 累加器清零
    float acc[N8][4]={0.0f};
    #pragma unroll
    // for (int n = 0; n < N8; ++n) {
    //     for (int j = 0; j < 4; ++j) acc[n][j] = 0.0f;
    // }

    const int n_kb = K / TILE_K;
    const uint32_t txA = static_cast<uint32_t>(a_stage);
    const uint32_t txB = static_cast<uint32_t>(b_stage);
    bool phase[NUM_STAGE] = {false, false};

    // -------------------------------------------------------------------------
    // 预热 (Prologue): 生产者 (Thread 0) 发射 Stage 0 的 TMA 请求
    // -------------------------------------------------------------------------
    if (tid == 0) {
        mbar_ex(mb[0], txA + txB);
        tma2d(ap[0], mapA, 0, bm, mb[0]);  // A: {coord0=K_offset(0), coord1=M_offset(bm)}
        tma2d(bp[0], mapB, bn, 0, mb[0]);  // B: {coord0=N_offset(bn), coord1=K_offset(0)}
        //通过CUtensorMap判断提取大小
    }

    // -------------------------------------------------------------------------
    // 主循环 (Main Loop): K 轴推进
    // -------------------------------------------------------------------------
    const int lane = tid % 32;
    const int lm_A = lane % 16;       // A 片段内的 16 行分配
    const int lc_A = (lane / 16) * 8; // A 片段内的 K 偏移 (0 或 8)

    for (int kb = 0; kb < n_kb; ++kb) {
        int sc = kb & 1; // 当前消费的 Stage
        int sn = sc ^ 1; // 下一预加载的 Stage

        // 1. 生产者预发下一个 Stage 的 TMA (实现计算与数据搬运重叠)
        if (tid == 0 && (kb + 1) < n_kb) {
            int knext = (kb + 1) * TILE_K;
            mbar_ex(mb[sn], txA + txB);
            tma2d(ap[sn], mapA, knext, bm, mb[sn]);
            tma2d(bp[sn], mapB, bn, knext, mb[sn]);
        }

        // 2. 消费者等待当前 Stage 数据全满到位
        mbar_wait(mb[sc], phase[sc] ? 1u : 0u);
        phase[sc] = !phase[sc]; // 翻转 Phase 供下次消费

        // 3. 计算 Block: 沿 K 展开
        #pragma unroll
        for (int kk = 0; kk < KS; ++kk) {
            int koff = kk * MMA_K;

            // ===== 加载 A 片段 (16x16) =====
            // 每个 Warp 处理 16 行，ldmatrix.x4 (no trans) 读取 Row-Major 片段
            //这句话极为重要
            int gm = warp * 16 + lm_A;
            int gk = koff + lc_A;
            uint32_t logicalA = static_cast<uint32_t>(gm * TILE_K + gk) * sizeof(half);
            //在当前128*64的tile的逻辑地址。
            uint32_t physA    = swizzle128(logicalA);

            unsigned a0, a1, a2, a3;
            ldmatrix_a_x4(a0, a1, a2, a3, ap[sc] + physA); // 修复: 正确传入 +physA 偏移

            // ===== 对每一个 N8 子块: 加载 B 片段 + 计算 =====
            #pragma unroll
            for (int n = 0; n < N8; ++n) {
                // 加载 B 片段 (16x8): ldmatrix.x2.trans 读入并转置为 Col-Major 寄存器布局
                int kb_row = koff + (lane % 16);
                int colB   = n * 8;
                uint32_t logicalB = static_cast<uint32_t>(kb_row * TILE_N + colB) * sizeof(half);
                uint32_t physB    = swizzle128(logicalB);

                unsigned b0, b1;
                ldmatrix_b_x2_trans(b0, b1, bp[sc] + physB);

                // PTX mma.sync 执行
                mma_sync_m16n8k16(
                    acc[n][0], acc[n][1], acc[n][2], acc[n][3],
                    a0, a1, a2, a3,
                    b0, b1
                );
            }
        }

        // 保证当前 Stage Shared Memory 彻底消费完毕，再进入下一次 Iteration
        __syncthreads();
    }

    // -------------------------------------------------------------------------
    // 写回 C 矩阵 (FP32 行主序 Global Memory)
    // -------------------------------------------------------------------------
    const int rA = lane / 4;          // D 片段标准映射：行
    const int cA = (lane % 4) * 2;      // D 片段标准映射：列

    #pragma unroll
    for (int n = 0; n < N8; ++n) {
        int row = bm + warp * 16 + rA;
        int col = bn + n * 8 + cA;

        if (row < M && (col + 1) < N) {
            dC[static_cast<size_t>(row) * N + col]         = acc[n][0]; // (r, c)
            dC[static_cast<size_t>(row) * N + col + 1]     = acc[n][1]; // (r, c+1)
            dC[static_cast<size_t>(row + 8) * N + col]     = acc[n][2]; // (r+8, c)
            dC[static_cast<size_t>(row + 8) * N + col + 1] = acc[n][3]; // (r+8, c+1)0
            
        }
    }
}

// -----------------------------------------------------------------------------
// Host 侧代码
// -----------------------------------------------------------------------------
int main() {
    CUK(cuInit(0));
    CUdevice dv;
    CUcontext ctx;
    CUK(cuDeviceGet(&dv, 0));
    CUK(cuCtxCreate(&ctx, 0, dv));

    const int M = 1024, N = 4096, K = 4096;
    size_t smem = NUM_STAGE * (TILE_M * TILE_K + TILE_K * TILE_N) * sizeof(half) + NUM_STAGE * sizeof(uint64_t);

    CUDA_CHECK(cudaFuncSetAttribute(gemm_ptx, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));

    std::srand(2026);
    std::vector<half> hA(static_cast<size_t>(M) * K), hB(static_cast<size_t>(K) * N);
    for (auto& x : hA) x = __float2half(static_cast<float>(std::rand() % 7 - 3));
    for (auto& x : hB) x = __float2half(static_cast<float>(std::rand() % 7 - 3));

    half *dA, *dB;
    float *dC;
    CUDA_CHECK(cudaMalloc(&dA, static_cast<size_t>(M) * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dB, static_cast<size_t>(K) * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dC, static_cast<size_t>(M) * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), static_cast<size_t>(M) * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), static_cast<size_t>(K) * N * sizeof(half), cudaMemcpyHostToDevice));

    // TMA TensorMap 编码配置
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

    CUtensorMap *dma, *dmb;
    CUDA_CHECK(cudaMalloc(&dma, sizeof(CUtensorMap)));
    CUDA_CHECK(cudaMalloc(&dmb, sizeof(CUtensorMap)));
    CUDA_CHECK(cudaMemcpy(dma, &mapA, sizeof(CUtensorMap), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dmb, &mapB, sizeof(CUtensorMap), cudaMemcpyHostToDevice));

    dim3 grid(N / TILE_N, M / TILE_M);
    dim3 block(256);
    printf("grid=(%d,%d) block=256 smem=%zu (TILE 128x64, 双缓冲, SWIZZLE_128B)\n", grid.x, grid.y, smem);

    // Warmup
    for (int i = 0; i < 3; i++) gemm_ptx<<<grid, block, smem>>>(dma, dmb, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 性能基准测试
    double best = 1e9;
    for (int t = 0; t < 5; t++) {
        cudaEvent_t s, e;
        cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        gemm_ptx<<<grid, block, smem>>>(dma, dmb, dC, M, N, K);
        cudaEventRecord(e);
        cudaGetLastError();
        cudaDeviceSynchronize();
        cudaEventSynchronize(e);

        float ms = 0;
        cudaEventElapsedTime(&ms, s, e);
        if (ms < best) best = ms;
        cudaEventDestroy(s); cudaEventDestroy(e);
    }
    double flops = 2.0 * static_cast<double>(M) * N * K;
    printf("Time=%.4f ms | Compute=%.2f TFLOPS\n", best, flops / (best * 1e9));

    // -------------------------------------------------------------------------
    // CPU 数值校验 (抽检 128x128 区域)
    // -------------------------------------------------------------------------
    std::vector<float> hC(static_cast<size_t>(M) * N);
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, static_cast<size_t>(M) * N * sizeof(float), cudaMemcpyDeviceToHost));

    double max_diff = 0.0, sum_diff = 0.0;
    int check_cnt = 0;
    for (int m = 0; m < 128; ++m) {
        for (int n = 0; n < 128; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += __half2float(hA[static_cast<size_t>(m) * K + k]) *
                       __half2float(hB[static_cast<size_t>(k) * N + n]);
            }
            double diff = std::fabs(static_cast<double>(hC[static_cast<size_t>(m) * N + n]) - acc);
            if (diff > max_diff) max_diff = diff;
            sum_diff += diff;
            ++check_cnt;
        }
    }

    double tol = 1e-2; // FP32 累加精度容差
    printf("CPU Check: Max Error = %.4f | Mean Error = %.6f | Status = %s\n",
           max_diff, sum_diff / check_cnt, (max_diff < tol) ? "[PASS]" : "[FAIL]");

    return (max_diff < tol) ? 0 : 1;
}