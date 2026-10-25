#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <cstdlib>
#include <cstdio>

// CUDA 运行时 API 头文件
#include <cuda_runtime.h>

// cuBLAS API 头文件
#include <cublas_v2.h>

// ============================================================================
// 1. CUDA Runtime API 错误检查宏
// ============================================================================
#define CHECK_CUDA(call) \
    do { \
        const cudaError_t error_code = (call); \
        if (error_code != cudaSuccess) { \
            std::fprintf(stderr, "[CUDA ERROR] at %s:%d -> %s: %s\n", \
                         __FILE__, __LINE__, \
                         cudaGetErrorName(error_code), \
                         cudaGetErrorString(error_code)); \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

// ============================================================================
// 2. cuBLAS API 错误检查宏与辅助转换函数
// ============================================================================

// 将 cuBLAS 枚举状态码转换为人类可读字符串的辅助函数
inline const char* _cublasGetErrorEnum(cublasStatus_t error) {
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:          return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:     return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:    return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:    return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:    return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:   return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:    return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:    return "CUBLAS_STATUS_LICENSE_ERROR";
        default:                             return "<unknown>";
    }
}

#define CHECK_CUBLAS(call) \
    do { \
        const cublasStatus_t status = (call); \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::fprintf(stderr, "[cuBLAS ERROR] at %s:%d -> %s\n", \
                         __FILE__, __LINE__, \
                         _cublasGetErrorEnum(status)); \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

// ============================================================================
// 3. (可选扩展) Kernel 异步启动错误检查宏
// ============================================================================
// 专门用于在自己手写 <<<grid, block>>> 异步核函数之后，捕捉核函数内部爆出的错误
#define CHECK_LAST_CUDA_ERROR() \
    do { \
        const cudaError_t error_code = cudaGetLastError(); \
        if (error_code != cudaSuccess) { \
            std::fprintf(stderr, "[CUDA KERNEL ERROR] at %s:%d -> %s: %s\n", \
                         __FILE__, __LINE__, \
                         cudaGetErrorName(error_code), \
                         cudaGetErrorString(error_code)); \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

#endif // COMMON_H