#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// ---------- 错误检查宏 ----------
// 用法：CUDA_CHECK(cudaMalloc(...));
// 失败时打印哪个文件、哪一行、什么错误，然后退出
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            printf("CUDA 错误：%s  (%s:%d)\n",                            \
                   cudaGetErrorString(err), __FILE__, __LINE__);          \
            exit(1);                                                      \
        }                                                                 \
    } while (0)