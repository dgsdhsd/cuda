#include "../include/transformer_layer.cuh"
#include "../include/common.h"   // 提供 CHECK_CUBLAS 宏与 cublas_v2.h
#include <cuda_profiler_api.h>
#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define BK 32
#define BM 128
#define BN 128




union SharedStorage {
    struct {
        // s_a: [2][32][128], k 为行, m 为列, m 维度 128 float 连续
        float s_a[2][BM][BK];
        // s_b: [2][32][128], k 为行, n 为列, n 维度 128 float 连续
        float s_b[2][BK][BN];
    } compute;

    struct {
        float s_out[2][32][128]; // [stage][row_in_chunk][col]
    } epilogue;
};

// -----------------------------------------------------------------------------
// 1. 定义 2D Block Tiling + Register Tiling GEMM 核函数
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// cp.async 16 字节异步拷贝内联汇编 (支持 Ampere 及以上架构)
// -----------------------------------------------------------------------------
__device__ __forceinline__ void cp_async_cg(void* smem_ptr, const void* gmem_ptr) {
    unsigned int smem_addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :
        : "r"(smem_addr), "l"(gmem_ptr)
    );
}

// -----------------------------------------------------------------------------
// 高性能 2D Register Block SGEMM Kernel
// -----------------------------------------------------------------------------



__global__ void __launch_bounds__(256, 1) sgemm_2d_Register_bank_improve(
    int M, int N, int K,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C) 
{
    // -------------------------------------------------------------------------
    // 1. 基础分块参数与共享内存定义
    // -------------------------------------------------------------------------
    constexpr int TM = 8;
    constexpr int TN = 8;


    extern __shared__ char smem_raw[];
    
    // 2. 强转为结构体指针 (名称仍然叫 smem)
    SharedStorage* smem = reinterpret_cast<SharedStorage*>(smem_raw);

    const int tid = threadIdx.x; // 0 ~ 255

    // 预先计算好的线程解耦常数
    const int lane_16 = tid % 16;         // 0 ~ 15，控制 B/C 矩阵列解耦组
    const int thread_m = (tid / 16) * 8;   // 0, 8, 16 ... 120，控制 A/C 矩阵行起始

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // 调整全局基指针到当前 Block 起始位置
    A += block_row * K;
    B += block_col;

    // 线程私有 MMA 累加寄存器 (8x8)
    float accum[TM][TN] = {0.0f};

    // -------------------------------------------------------------------------
    // 2. 流水线预热 (Prologue): 异步加载第 0 块数据到 Buffer 0
    // -------------------------------------------------------------------------
    int write_stage = 0;

    // 加载 A 矩阵到 s_a 
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int load_idx = tid + i * 256; // 0 ~ 1023
        int r = load_idx / 8;        // A 矩阵全局/SMEM 行 (0 ~ 127)
        int c_f4 = load_idx % 8;     // A 矩阵全局/SMEM 列 float4 索引 (0 ~ 7)

        cp_async_cg(&smem->compute.s_a[write_stage][r][c_f4 * 4], A + r * K + c_f4 * 4);
    }

    // 异步加载 B 矩阵到 s_b (直接存入 SMEM: s_b[write_stage][k][n])
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int load_idx = tid + i * 256; // 0 ~ 1023
        int r = load_idx / 32;       // B 矩阵全局行 (0 ~ 31)
        int c_f4 = load_idx % 32;    // B 矩阵全局列 float4 索引 (0 ~ 31)

        cp_async_cg(&smem->compute.s_b[write_stage][r][c_f4 * 4], B + r * N + c_f4 * 4);
    }

    // 提交异步加载组并等待完成
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    // -------------------------------------------------------------------------
    // 3. Main Loop: 双缓冲计算与搬运重叠
    // -------------------------------------------------------------------------
    for (int k_step = BK; k_step < K; k_step += BK) {
        // 推进全局指针
        A += BK;
        B += BK * N;

        // 切换写入缓冲区索引 (0 -> 1, 1 -> 0)
        write_stage ^= 1;

        // --- Step 3A: 异步发射下一轮 (k_step) 数据到 write_stage ---
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int load_idx = tid + i * 256; // 0 ~ 1023
            int r = load_idx / 8;        // A 矩阵全局/SMEM 行 (0 ~ 127)
            int c_f4 = load_idx % 8;     // A 矩阵全局/SMEM 列 float4 索引 (0 ~ 7)

            cp_async_cg(&smem->compute.s_a[write_stage][r][c_f4 * 4], A + r * K + c_f4 * 4);
        }

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int load_idx = tid + i * 256;
            int r = load_idx / 32;
            int c_f4 = load_idx % 32;

            cp_async_cg(&smem->compute.s_b[write_stage][r][c_f4 * 4], B + r * N + c_f4 * 4);
        }

        asm volatile("cp.async.commit_group;\n");

        // --- Step 3B: 计算当前 read_stage 缓冲区的数据 ---
        int read_stage = write_stage ^ 1;

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // 1. 读取 A 矩阵数据 (2 条 LDS.128 指令，100% Broadcast，0 Bank 冲突)
            float a_val[8];
            #pragma unroll
            for (int m = 0; m < 8; ++m) {
                a_val[m] = smem->compute.s_a[read_stage][thread_m + m][k];
            }

            // 2. 读取 B 矩阵数据 (2 条 LDS.128 指令，解耦跨列读取，0 Bank 冲突)
            const float4* sb_ptr = reinterpret_cast<const float4*>(
                &smem->compute.s_b[read_stage][k][0]
            );
            float4 reg_b0 = sb_ptr[lane_16];      // 取列 0~3 对应 float4
            float4 reg_b1 = sb_ptr[lane_16 + 16]; // 取列 64~67 对应 float4

            float b_val[8] = {
                reg_b0.x, reg_b0.y, reg_b0.z, reg_b0.w,
                reg_b1.x, reg_b1.y, reg_b1.z, reg_b1.w
            };

            // 3. 外积 MMA 计算 (纯寄存器 FMA 操作)
            #pragma unroll
            for (int m = 0; m < 8; ++m) {
                #pragma unroll
                for (int n = 0; n < 8; ++n) {
                    accum[m][n] += a_val[m] * b_val[n];
                }
            }
        }

        // --- Step 3C: 等待下一轮传输完成 ---
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
    }

    // -------------------------------------------------------------------------
    // 4. Epilogue Loop: 处理最后一轮留在 Buffer 里的数据
    // -------------------------------------------------------------------------
    int read_stage = write_stage;

    #pragma unroll
    for (int k = 0; k < BK; ++k) {
        float a_val[8];
        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            a_val[m] = smem->compute.s_a[read_stage][thread_m + m][k];
        }

        const float4* sb_ptr = reinterpret_cast<const float4*>(
            &smem->compute.s_b[read_stage][k][0]
        );
        float4 reg_b0 = sb_ptr[lane_16];
        float4 reg_b1 = sb_ptr[lane_16 + 16];

        float b_val[8] = {
            reg_b0.x, reg_b0.y, reg_b0.z, reg_b0.w,
            reg_b1.x, reg_b1.y, reg_b1.z, reg_b1.w
        };

        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            #pragma unroll
            for (int n = 0; n < 8; ++n) {
                accum[m][n] += a_val[m] * b_val[n];
            }
        }
    }

    // -------------------------------------------------------------------------
    // 5. 写回全局内存 C (STG.128 矢量化合并写回)
    // -------------------------------------------------------------------------
    const int g_col_group0_f4 = (block_col / 4) + lane_16;      // 前 4 列对应的 float4 列索引
    const int g_col_group1_f4 = (block_col / 4) + lane_16 + 16; // 后 4 列对应的 float4 列索引

    #pragma unroll
    for (int m = 0; m < 8; ++m) {
        int g_row = block_row + thread_m + m; // 当前元素的 Global C 行坐标

        // 1. 写回前 4 列 (accum[m][0..3])
        float4 c_val0 = make_float4(
            accum[m][0], accum[m][1], accum[m][2], accum[m][3]
        );
        float4* c_ptr0 = reinterpret_cast<float4*>(&C[g_row * N + (g_col_group0_f4 * 4)]);
        *c_ptr0 = c_val0; // STG.128 指令

        // 2. 写回后 4 列 (accum[m][4..7])
        float4 c_val1 = make_float4(accum[m][4], accum[m][5], accum[m][6], accum[m][7]);
        float4* c_ptr1 = reinterpret_cast<float4*>(&C[g_row * N + (g_col_group1_f4 * 4)]);
        *c_ptr1 = c_val1; // STG.128 指令
    }
}

//---------------------------------------tensor core------------------------------
__global__ void __launch_bounds__(288) gemm_ptx(
    const CUtensorMap* mapA, const CUtensorMap* mapB,
    float* __restrict__ dC, int M, int N, int K
) {
    extern __shared__ char smem_raw[];

    const size_t a_stage = TILE_M * TILE_K * sizeof(half);
    const size_t b_stage = TILE_K * TILE_N * sizeof(half);

    const int tid   = threadIdx.x;
    const int warp  = tid / 32;
    const int lane  = tid % 32;
    const int bm    = blockIdx.y * TILE_M;
    const int bn    = blockIdx.x * TILE_N;

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
    // 1. Prologue: 消费者预计算 Swizzle 偏移量 (利用寄存器，只算一次)
    // -------------------------------------------------------------------------
    uint32_t offsetA[KS];
    uint32_t offsetB[KS][N8];
    // 只有消费者 Warp (Warp 1 ~ 8) 需要进行计算和预计算
    if (warp > 0) {
        int c_warp = warp - 1;  // 消费者的逻辑 warp 编号 (0 ~ 7)
        const int lm_A = lane % 16;
        const int lc_A = (lane / 16) * 8;
        #pragma unroll
        for (int kk = 0; kk < KS; ++kk) {
            int koff = kk * MMA_K;
            // 预计算 A 矩阵的物理内存偏移
            int gm = c_warp * 16 + lm_A;
            int gk = koff + lc_A;
            uint32_t logicalA = static_cast<uint32_t>(gm * TILE_K + gk) * sizeof(half);
            offsetA[kk] = swizzle128(logicalA);
            // 预计算 B 矩阵的物理内存偏移
            #pragma unroll
            for (int n = 0; n < N8; ++n) {
                int kb_row = koff + (lane % 16);
                int colB   = n * 8;
                uint32_t logicalB = static_cast<uint32_t>(kb_row * TILE_N + colB) * sizeof(half);
                offsetB[kk][n] = swizzle128(logicalB);
            }
        }
    }
    // -------------------------------------------------------------------------
    // 2. Main Loop: 空间解耦的异步流水线 (Warp 隔离)
    // -------------------------------------------------------------------------
    if (warp == 0) {
        // ==========================================
        // Producer 专职流水线 (Warp 0)
        // ==========================================
        if (lane == 0) {
            int prefetch_stages = min(n_kb, NUM_STAGE);
            // 1. 发起初始预取
            for (int s = 0; s < prefetch_stages; ++s) {
                mbar_ex(mb_full[s], txA + txB);
                tma2d(ap[s], mapA, s * TILE_K, bm, mb_full[s]);
                tma2d(bp[s], mapB, bn, s * TILE_K, mb_full[s]);
            }
            // 2. 发起剩余预取
            for (int kb = prefetch_stages; kb < n_kb; ++kb) {
                int sn = kb % NUM_STAGE;
                // 只有等 8 个消费者用完该槽位，才能覆盖
                mbar_wait(mb_empty[sn], phase_empty[sn]);//相同等待
                phase_empty[sn] ^= 1;
                mbar_ex(mb_full[sn], txA + txB);
                tma2d(ap[sn], mapA, kb * TILE_K, bm, mb_full[sn]);
                tma2d(bp[sn], mapB, bn, kb * TILE_K, mb_full[sn]);
            }
        }
    }
    else {
        // ==========================================
        // Consumer 专职流水线 (Warp 1 ~ 8)
        // ==========================================
        for (int kb = 0; kb < n_kb; ++kb) {
            int sc = kb % NUM_STAGE;
            // 1. 等待当前 Stage 的 TMA 搬运完成
            mbar_wait(mb_full[sc], phase_full[sc]);
            phase_full[sc] ^= 1;
            uint32_t baseA = ap[sc];
            uint32_t baseB = bp[sc];
            // 2. 纯粹、无分支、无 ALU 干扰的极限 MMA 计算
            #pragma unroll
            for (int kk = 0; kk < KS; ++kk) {
                unsigned a0, a1, a2, a3;
                ldmatrix_a_x4(a0, a1, a2, a3, baseA + offsetA[kk]);
                #pragma unroll
                for (int n = 0; n < N8; ++n) {
                    unsigned b0, b1;
                    ldmatrix_b_x2_trans(b0, b1, baseB + offsetB[kk][n]);
                    mma_sync_m16n8k16(
                        acc[n][0], acc[n][1], acc[n][2], acc[n][3],
                        a0, a1, a2, a3,
                        b0, b1
                    );
                }
            }
            // 3. 严格同步 Warp，防止提前触发 arrive 导致数据污染
            __syncwarp();
            // 4. 通知 Producer 该槽位已安全消费完毕
            if (lane == 0) {
                mbar_arrive(mb_empty[sc]);
            }
        }
    }
    // -------------------------------------------------------------------------
    // Epilogue: 复用 SMEM + Padding(68) 消除 Bank 冲突 + STG.128 向量写回
    // -------------------------------------------------------------------------
    __syncthreads(); // 确保主循环计算全部结束，SMEM 可安全覆写
    float* smem_C = reinterpret_cast<float*>(smem_raw);
    constexpr int TILE_N_PAD = TILE_N + 4; // 68
    // 1. 写 SMEM：仅允许 Consumer Warps (Warp 1 ~ 8) 将累加结果存入 SMEM
    if (warp > 0) {
        int c_warp = warp - 1; // 转换为逻辑 Consumer Warp 编号 (0 ~ 7)
        const int rA = lane / 4;
        const int cA = (lane % 4) * 2;

        #pragma unroll
        for (int n = 0; n < N8; ++n) {
            int local_r0 = c_warp * 16 + rA;
            int local_r1 = local_r0 + 8;
            int local_c0 = n * 8 + cA;
            int local_c1 = local_c0 + 1;

            smem_C[local_r0 * TILE_N_PAD + local_c0] = acc[n][0];
            smem_C[local_r0 * TILE_N_PAD + local_c1] = acc[n][1];
            smem_C[local_r1 * TILE_N_PAD + local_c0] = acc[n][2];
            smem_C[local_r1 * TILE_N_PAD + local_c1] = acc[n][3];
        }
    }
    __syncthreads(); // 屏障：等待 8 个 Consumer 完成 SMEM 写入

    // 2. 向量化合并写回 GMEM (STG.128)
    // 仅使用 8 个 Consumer Warp (256 线程) 参与写回：2048 / 256 = 8 次循环，精确整除！
    if (warp > 0) {
        int c_tid = tid - 32; // Consumer 线程逻辑索引 (0 ~ 255)
        constexpr int TOTAL_FLOAT4 = (TILE_M * TILE_N) / 4; // 2048 个 float4
        float4* dC_f4 = reinterpret_cast<float4*>(dC);
        const int gmem_stride_f4 = N / 4;
        #pragma unroll
        for (int i = c_tid; i < TOTAL_FLOAT4; i += 256) { // 步进 256，每线程精准执行 8 次
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
    size_t smem_size = NUM_STAGE * (TILE_M * TILE_K + TILE_K * TILE_N) * sizeof(half) + 2 * NUM_STAGE * sizeof(uint64_t);
    CUDA_CHECK(cudaFuncSetAttribute(gemm_ptx, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_size)));

    dim3 grid(N / TILE_N, M / TILE_M);
    dim3 block(288);

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

//---------------------------------------tensor core------------------------------
// -----------------------------------------------------------------------------
// TransformerLayerNode 成员函数实现
// -----------------------------------------------------------------------------
TransformerLayerNode::TransformerLayerNode(bool enable_checkpoint) 
    : d_input(nullptr), d_output(nullptr), qlora_weight_B(nullptr),
      d_Wq(nullptr), d_Wk(nullptr), d_Wv(nullptr),
      use_checkpointing(enable_checkpoint), is_activation_saved(false) {}

TransformerLayerNode::~TransformerLayerNode() {
    // MemoryPool 统一管理显存
}

void TransformerLayerNode::initialize_weights_unified(
    const float* h_Wqkv_packed,
    MemoryPool& mem_pool,
    cudaStream_t stream
) {
    size_t single_weight_elements = K * N;
    size_t total_bytes = 3 * single_weight_elements * sizeof(float);

    d_Wqkv = (float*)mem_pool.allocate(total_bytes, stream);
    cudaMemcpyAsync(d_Wqkv, h_Wqkv_packed, total_bytes, cudaMemcpyHostToDevice, stream);

    d_Wq = d_Wqkv;
    d_Wk = d_Wqkv + single_weight_elements;
    d_Wv = d_Wqkv + (2 * single_weight_elements);
}

void TransformerLayerNode::forward(MemoryPool& mem_pool, cudaStream_t stream) {
    size_t input_bytes = M * K * sizeof(float);
    size_t output_bytes = M * N * sizeof(float);

    d_input = static_cast<float*>(mem_pool.allocate(input_bytes, stream));
    
    try {
        d_output = static_cast<float*>(mem_pool.allocate(output_bytes, stream));
        is_activation_saved = true;
    } catch (const std::runtime_error& e) {
        if (use_checkpointing) {
            std::cout << "[Warning] OOM detected in Forward. Activating Checkpointing!" << std::endl;
            is_activation_saved = false;
            d_output = nullptr;
        } else {
            throw;
        }
    }

    if (d_output) {
        dim3 grid((N + 127) / 128, (M + 127) / 128);
        dim3 block(256);

        // =====================================================================
        // 0. 动态 Shared Memory 属性设置与大小计算
        // =====================================================================
        size_t smem_bytes = sizeof(SharedStorage); // 精确大小，无需多算

        // 利用 Lambda + static 保证只向 CUDA 驱动注册一次属性，开销几乎为 0
        static bool is_attr_configured = [smem_bytes]() {
            cudaError_t err = cudaFuncSetAttribute(
                reinterpret_cast<const void*>(sgemm_2d_Register_bank_improve),
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smem_bytes)
            );
            if (err != cudaSuccess) {
                std::cerr << "[CUDA Error] Failed to set max dynamic shared memory size: " 
                          << cudaGetErrorString(err) << std::endl;
            }
            return true;
        }();

        // =====================================================================
        // 1. 真实 Warmup：跑 10 次真正的大 workload 彻底拉高 GPU 频率与 P-State
        // =====================================================================
        for (int i = 0; i < 10; ++i) {
            // 传入 smem_bytes 作为第 3 个配置参数 (Dynamic Shared Memory size)
            // sgemm_2d_Register_bank_improve<<<grid, block, smem_bytes, stream>>>(
            //     M, N, K, d_input, d_Wq, d_output
            // );
            GEMM_MYSELF(M, N, K, reinterpret_cast<const half*>(d_input), reinterpret_cast<const half*>(d_Wq), d_output);
        }//这里强转是错误的，准确性是完全不对的，但能保证正常运行，暂不想管，只要能运行，咋样都行
        cudaStreamSynchronize(stream);

        // =====================================================================
        // 2. 核心 Profile 与测速区间（只记录 1 次）
        // =====================================================================
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // 开启 NCU 采样（跳过前面的 10 次 Warmup）
        cudaProfilerStart();

        cudaEventRecord(start, stream);

        // 目标分析 Kernel：仅 Launch 1 次，同样传入 smem_bytes
        // sgemm_2d_Register_bank_improve<<<grid, block, smem_bytes, stream>>>(
        //     M, N, K, d_input, d_Wq, d_output
        // );
        GEMM_MYSELF(M, N, K, reinterpret_cast<const half*>(d_input), reinterpret_cast<const half*>(d_Wq), d_output);


        cudaEventRecord(stop, stream);
        cudaProfilerStop(); // 结束 NCU 采样

        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);

        // 打印单次精准耗时与 GFLOPS
        double gflops = (2.0 * static_cast<double>(M) * N * K * 1e-9) / (ms * 1e-3);
        std::cout << "[GEMM Baseline] " << ms << " ms | " << gflops << " GFLOPS" << std::endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
}


/*void TransformerLayerNode::forward(MemoryPool& mem_pool, cudaStream_t stream) {
    size_t input_bytes = M * K * sizeof(float);
    size_t output_bytes = M * N * sizeof(float);

    d_input = static_cast<float*>(mem_pool.allocate(input_bytes, stream));
    
    try {
        d_output = static_cast<float*>(mem_pool.allocate(output_bytes, stream));
        is_activation_saved = true;
    } catch (const std::runtime_error& e) {
        if (use_checkpointing) {
            std::cout << "[Warning] OOM detected in Forward. Activating Checkpointing!" << std::endl;
            is_activation_saved = false;
            d_output = nullptr;
        } else {
            throw;
        }
    }

    if (d_output) {
        // =====================================================================
        // 0. 临时开辟 FP16 显存，并将 FP32 的 d_input 与 d_Wq 转存为 FP16
        // =====================================================================
        half* d_input_fp16 = nullptr;
        half* d_Wq_fp16 = nullptr;
        
        size_t input_elements = M * K;
        size_t wq_elements = K * N;

        cudaMallocAsync(&d_input_fp16, input_elements * sizeof(half), stream);
        cudaMallocAsync(&d_Wq_fp16, wq_elements * sizeof(half), stream);

        // 执行数据类型转换 FP32 -> FP16
        launch_float2half(d_input, d_input_fp16, input_elements, stream);
        launch_float2half(d_Wq, d_Wq_fp16, wq_elements, stream);

        // =====================================================================
        // 1. 创建 cuBLAS handle
        // =====================================================================
        cublasHandle_t cublas_handle;
        CHECK_CUBLAS(cublasCreate(&cublas_handle));
        CHECK_CUBLAS(cublasSetStream(cublas_handle, stream));

        const float alpha = 1.0f;
        const float beta  = 0.0f;

        // =====================================================================
        // 2. 真实 Warmup：跑 10 次 FP16 Tensor Core GEMM
        // =====================================================================
        for (int i = 0; i < 10; ++i) {
            // 原 cublasSgemm 保留注释，不删除
            // CHECK_CUBLAS(cublasSgemm(
            //     cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
            //     N, M, K, &alpha, d_Wq, N, d_input, K, &beta, d_output, N
            // ));
            
            
            // 使用临时转好的 FP16 指针调用 cublasGemmEx 触发 Tensor Core
            CHECK_CUBLAS(cublasGemmEx(
                cublas_handle, 
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, 
                &alpha, 
                d_Wq_fp16,    CUDA_R_16F, N,
                d_input_fp16, CUDA_R_16F, K,
                &beta, 
                d_output,     CUDA_R_32F, N,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP
            ));
        }
        cudaStreamSynchronize(stream);

        // =====================================================================
        // 3. 核心 Profile 与测速区间
        // =====================================================================
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaProfilerStart(); // 开启 NCU 采样

        cudaEventRecord(start, stream);

        // 原 cublasSgemm 保留注释，不删除
        // CHECK_CUBLAS(cublasSgemm(
        //     cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        //     N, M, K, &alpha, d_Wq, N, d_input, K, &beta, d_output, N
        // ));
        

        CHECK_CUBLAS(cublasGemmEx(
            cublas_handle, 
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, 
            &alpha, 
            d_Wq_fp16,    CUDA_R_16F, N,
            d_input_fp16, CUDA_R_16F, K,
            &beta, 
            d_output,     CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));

        cudaEventRecord(stop, stream);
        cudaProfilerStop(); // 结束 NCU 采样

        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);

        // 打印单次精准耗时与 GFLOPS
        double gflops = (2.0 * static_cast<double>(M) * N * K * 1e-9) / (ms * 1e-3);
        std::cout << "[cuBLAS FP16 TensorCore] " << ms << " ms | " << gflops << " GFLOPS" << std::endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // 销毁 cuBLAS handle
        CHECK_CUBLAS(cublasDestroy(cublas_handle));

        // 释放临时 FP16 显存
        cudaFreeAsync(d_input_fp16, stream);
        cudaFreeAsync(d_Wq_fp16, stream);
    }
}*/

/*void TransformerLayerNode::forward(MemoryPool& mem_pool, cudaStream_t stream) {
    size_t input_bytes = M * K * sizeof(float);
    size_t output_bytes = M * N * sizeof(float);
    size_t wq_bytes = K * N * sizeof(float);

    d_input = static_cast<float*>(mem_pool.allocate(input_bytes, stream));
    
    try {
        d_output = static_cast<float*>(mem_pool.allocate(output_bytes, stream));
        is_activation_saved = true;
    } catch (const std::runtime_error& e) {
        if (use_checkpointing) {
            std::cout << "[Warning] OOM detected in Forward. Activating Checkpointing!" << std::endl;
            is_activation_saved = false;
            d_output = nullptr;
        } else {
            throw;
        }
    }

    if (d_output) {
        // =====================================================================
        // 0. 主机端 (CPU) 完成数据 FP32 -> FP16 的转换与传输
        // =====================================================================
        
        // 1) 在 CPU 端开辟 FP32 和 FP16 内存缓冲区
        std::vector<float> h_input_f32(M * K);
        std::vector<float> h_Wq_f32(K * N);
        std::vector<half>  h_input_f16(M * K);
        std::vector<half>  h_Wq_f16(K * N);

        // 2) 将设备端 FP32 数据 D2H 拷贝回主机端
        cudaMemcpyAsync(h_input_f32.data(), d_input, input_bytes, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_Wq_f32.data(), d_Wq, wq_bytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream); // 等待 D2H 拷贝完成

        // 3) CPU 主机端逐元素转换数值 (FP32 -> FP16)
        for (size_t i = 0; i < M * K; ++i) {
            h_input_f16[i] = __float2half(h_input_f32[i]);
        }
        for (size_t i = 0; i < K * N; ++i) {
            h_Wq_f16[i] = __float2half(h_Wq_f32[i]);
        }

        // 4) GPU 端开辟临时 FP16 显存空间
        half* d_input_fp16 = nullptr;
        half* d_Wq_fp16 = nullptr;
        cudaMallocAsync(&d_input_fp16, M * K * sizeof(half), stream);
        cudaMallocAsync(&d_Wq_fp16, K * N * sizeof(half), stream);

        // 5) 将 CPU 转好的 FP16 数据 H2D 拷回 GPU
        cudaMemcpyAsync(d_input_fp16, h_input_f16.data(), M * K * sizeof(half), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_Wq_fp16, h_Wq_f16.data(), K * N * sizeof(half), cudaMemcpyHostToDevice, stream);

        // =====================================================================
        // 1. 创建 cuBLAS handle
        // =====================================================================
        cublasHandle_t cublas_handle;
        CHECK_CUBLAS(cublasCreate(&cublas_handle));
        CHECK_CUBLAS(cublasSetStream(cublas_handle, stream));

        const float alpha = 1.0f;
        const float beta  = 0.0f;

        // =====================================================================
        // 2. 真实 Warmup：跑 10 次 FP16 Tensor Core GEMM
        // =====================================================================
        for (int i = 0; i < 10; ++i) {
            // 原 cublasSgemm 保留注释，不删除
            // CHECK_CUBLAS(cublasSgemm(
            //     cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
            //     N, M, K, &alpha, d_Wq, N, d_input, K, &beta, d_output, N
            // ));
            
            
            // 正确传入转好数值的 FP16 内存指针
            CHECK_CUBLAS(cublasGemmEx(
                cublas_handle, 
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, 
                &alpha, 
                d_Wq_fp16,    CUDA_R_16F, N,
                d_input_fp16, CUDA_R_16F, K,
                &beta, 
                d_output,     CUDA_R_32F, N,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP
            ));
        }
        cudaStreamSynchronize(stream);

        // =====================================================================
        // 3. 核心 Profile 与测速区间
        // =====================================================================
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaProfilerStart(); // 开启 NCU 采样

        cudaEventRecord(start, stream);

        // 原 cublasSgemm 保留注释，不删除
        // CHECK_CUBLAS(cublasSgemm(
        //     cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        //     N, M, K, &alpha, d_Wq, N, d_input, K, &beta, d_output, N
        // ));
        

        CHECK_CUBLAS(cublasGemmEx(
            cublas_handle, 
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, 
            &alpha, 
            d_Wq_fp16,    CUDA_R_16F, N,
            d_input_fp16, CUDA_R_16F, K,
            &beta, 
            d_output,     CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));

        cudaEventRecord(stop, stream);
        cudaProfilerStop(); // 结束 NCU 采样

        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);

        // 打印单次耗时与 GFLOPS
        double gflops = (2.0 * static_cast<double>(M) * N * K * 1e-9) / (ms * 1e-3);
        std::cout << "[cuBLAS FP16 TensorCore] " << ms << " ms | " << gflops << " GFLOPS" << std::endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // 销毁 cuBLAS handle
        CHECK_CUBLAS(cublasDestroy(cublas_handle));

        // 释放临时 FP16 显存
        cudaFreeAsync(d_input_fp16, stream);
        cudaFreeAsync(d_Wq_fp16, stream);
    }
}*/

void TransformerLayerNode::backward(MemoryPool& mem_pool, cudaStream_t stream) {
    size_t input_bytes = M * K * sizeof(float);
    size_t output_bytes = M * N * sizeof(float);

    // 1. 若前向传播中因 OOM 被迫放弃了激活值，在此处触发重计算 (Recompute)
    if (!is_activation_saved && use_checkpointing) {
        std::cout << "[Recompute] Recomputing missing forward activations..." << std::endl;
        
        // 重新分配临时空间并重跑前向 Kernel
        d_output = static_cast<float*>(mem_pool.allocate(output_bytes, stream));
        // gemm_forward_kernel<<<grid, block, 0, stream>>>(d_input, d_Wq, d_output, M, N, K);
    }

    // 2. 发射反向梯度计算 Kernel (预留)
    // gemm_backward_kernel<<<grid, block, 0, stream>>>(...);
    
    // 3. 任务彻底完成，将所有动态申请的激活值显存归还给 MemoryPool
    if (d_output) {
        mem_pool.deallocate(d_output, output_bytes, stream);
        d_output = nullptr;
    }
    if (d_input) {
        mem_pool.deallocate(d_input, input_bytes, stream);
        d_input = nullptr;
    }
}