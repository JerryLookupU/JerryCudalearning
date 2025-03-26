
__device__  __forceinline__ int ld_volatile_global(const int *ptr) {
    int ret;
    asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__  __forceinline__ float ld_volatile_global(const float *ptr) {
    float ret;
    asm volatile("ld.volatile.global.f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
    return ret;
}

__device__  __forceinline__ int64_t ld_volatile_global(const int64_t *ptr) {
    int64_t ret;
    asm volatile("ld.volatile.global.s64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

__device__  __forceinline__ int64_t ld_volatile_global(const uint64_t *ptr) {
    int64_t ret;
    asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}
//函数都是CUDA设备函数，用于执行volatile语义的全局内存加载操作。
// `ld_volatile_global` 系列函数针对不同数据类型(int/float/int64_t/uint64_t)实现了从全局内存加载数据的功能，
// 使用`volatile` 修饰确保编译器不会优化掉这些内存访问操作。每个函数都通过内联汇编精确控制PTX指令，其中int/float使用不同寄存器约束("=r"和"=f")，64位类型使用"=l"约束。

#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>

__global__ void test_ld_volatile_global(int* d_int, float* d_float, int64_t* d_int64, uint64_t* d_uint64) {
    int idx = threadIdx.x;
    if (idx == 0) {
        printf("Device: int=%d, float=%.2f, int64=%lld, uint64=%llu\n",
               ld_volatile_global(d_int),
               ld_volatile_global(d_float),
               ld_volatile_global(d_int64),
               ld_volatile_global(d_uint64));
    }
}

int main() {
    // 准备测试数据
    int h_int = 42;
    float h_float = 3.14f;
    int64_t h_int64 = 1234567890123456LL;
    uint64_t h_uint64 = 9876543210987654ULL;
    
    // 分配设备内存
    int* d_int;
    float* d_float;
    int64_t* d_int64;
    uint64_t* d_uint64;
    cudaMalloc(&d_int, sizeof(int));
    cudaMalloc(&d_float, sizeof(float));
    cudaMalloc(&d_int64, sizeof(int64_t));
    cudaMalloc(&d_uint64, sizeof(uint64_t));
    
    // 拷贝数据到设备
    cudaMemcpy(d_int, &h_int, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_float, &h_float, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_int64, &h_int64, sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uint64, &h_uint64, sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    // 启动测试内核
    test_ld_volatile_global<<<1, 1>>>(d_int, d_float, d_int64, d_uint64);
    cudaDeviceSynchronize();
    
    // 验证结果
    int r_int;
    float r_float;
    int64_t r_int64;
    uint64_t r_uint64;
    cudaMemcpy(&r_int, d_int, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&r_float, d_float, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&r_int64, d_int64, sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&r_uint64, d_uint64, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    assert(r_int == h_int);
    assert(r_float == h_float);
    assert(r_int64 == h_int64);
    assert(r_uint64 == h_uint64);
    
    printf("Host: int=%d, float=%.2f, int64=%lld, uint64=%llu\n",
           r_int, r_float, r_int64, r_uint64);
    
    // 释放资源
    cudaFree(d_int);
    cudaFree(d_float);
    cudaFree(d_int64);
    cudaFree(d_uint64);
    
    return 0;
}

