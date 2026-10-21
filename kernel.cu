// ============================================================
//  CUDA 12.8 Runtime 模板（精简版 + 中文注释）
//
//  原始模板 122 行，其中约 40 行是重复的 if (cudaStatus != cudaSuccess)。
//  用 CUDA_CHECK 宏替代后，真正的 CUDA 操作只剩 5 步。
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "common.cuh"
// device_launch_parameters.h 里声明了 threadIdx / blockIdx / blockDim / gridDim
// 它的作用是让 VS 的 IntelliSense（那个"错误列表"）认识这些名字，
// 不加的话会一直报 "未定义标识符 threadIdx"（E0020），但**编译其实是过的**。
#include <device_launch_parameters.h>



// ---------- 核函数：在 GPU 上跑的代码 ----------
// 每个线程负责一个元素：第 i 个线程算 c[i] = a[i] + b[i]
__global__ void addKernel(const int *a, const int *b, int *c, int size)
{
    int i = threadIdx.x;          // 本线程在本 block 内的编号
    if (i < size)                 // 越界保护：线程数可能多于元素数
        c[i] = a[i] + b[i];
}

int main()
{
    // ===== 主机端数据 =====
    const int size = 5;
    const int a[size] = {1, 2, 3, 4, 5};
    const int b[size] = {10, 20, 30, 40, 50};
    int c[size] = { 0 };

    // ===== 第 1 步：申请显存 =====
    int *dev_a, *dev_b, *dev_c;
    CUDA_CHECK(cudaMalloc(&dev_a, size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dev_b, size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dev_c, size * sizeof(int)));

    // ===== 第 2 步：把数据从主机内存拷到显存 =====
    CUDA_CHECK(cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice));

    // ===== 第 3 步：启动核函数 =====
    // <<<1, size>>> 读作：1 个 block，block 里 size 个线程
    addKernel<<<1, size>>>(dev_a, dev_b, dev_c, size);

    CUDA_CHECK(cudaGetLastError());        // 启动失败能立刻抓到
    CUDA_CHECK(cudaDeviceSynchronize());   // 等 GPU 真正算完（kernel 是异步的）

    // ===== 第 4 步：把结果从显存拷回主机 =====
    CUDA_CHECK(cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost));

    // ===== 第 5 步：释放显存 =====
    CUDA_CHECK(cudaFree(dev_a));
    CUDA_CHECK(cudaFree(dev_b));
    CUDA_CHECK(cudaFree(dev_c));

    // ===== 打印结果 =====
    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
           c[0], c[1], c[2], c[3], c[4]);

    // 重置设备：Nsight / Visual Profiler 需要它才能拿到完整 trace
    cudaDeviceReset();

    return 0;
}
