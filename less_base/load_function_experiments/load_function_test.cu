#include <cuda_runtime.h>
#include <stdio.h>

__device__ __forceinline__ int ld_acquire_cta(const int *ptr) {
    int ret;
    asm volatile("ld.acquire.cta.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}


// 各种类型绕过L1缓存直接访问内存方法
// 在CUDA PTX汇编中，ld.relaxed.gpu.global.L1::no_allocate指令支持的数据类型包括：8位(b8)、16位(b16)、32位(b32)和64位(b64)的无符号整数类型，分别对应uint8_t、uint16_t、uint32_t和uint64_t。
__device__ __forceinline__ uint8_t ld_na_relaxed(const uint8_t *ptr) {
    uint16_t ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b8 %0, [%1];" : "=h"(ret) : "l"(ptr));
    return static_cast<uint8_t>(ret);
}

__device__ __forceinline__ uint16_t ld_na_relaxed(const uint16_t *ptr) {
    uint16_t ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b16 %0, [%1];" : "=h"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ uint32_t ld_na_relaxed(const uint32_t *ptr) {
    uint32_t ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ uint64_t ld_na_relaxed(const uint64_t *ptr) {
    uint64_t ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

__global__ void test_ld_funcs(int *dev_int, uint8_t *dev_uint8, uint16_t *dev_uint16, uint32_t *dev_uint32, uint64_t *dev_uint64) {
    int result_cta = ld_acquire_cta(dev_int);
    printf("ld_acquire_cta(int) result: %d\n", result_cta);

    uint8_t result_uint8 = ld_na_relaxed(dev_uint8);
    printf("ld_na_relaxed(uint8_t) result: %u\n", result_uint8);

    uint16_t result_uint16 = ld_na_relaxed(dev_uint16);
    printf("ld_na_relaxed(uint16_t) result: %u\n", result_uint16);

    uint32_t result_uint32 = ld_na_relaxed(dev_uint32);
    printf("ld_na_relaxed(uint32_t) result: %u\n", result_uint32);

    uint64_t result_uint64 = ld_na_relaxed(dev_uint64);
    printf("ld_na_relaxed(uint64_t) result: %llu\n", result_uint64);
}

int main() {
    int host_int = 42;
    int *dev_int;
    cudaMalloc((void**)&dev_int, sizeof(int));
    cudaMemcpy(dev_int, &host_int, sizeof(int), cudaMemcpyHostToDevice);

    uint8_t host_uint8 = 8;
    uint8_t *dev_uint8;
    cudaMalloc((void**)&dev_uint8, sizeof(uint8_t));
    cudaMemcpy(dev_uint8, &host_uint8, sizeof(uint8_t), cudaMemcpyHostToDevice);

    uint16_t host_uint16 = 16;
    uint16_t *dev_uint16;
    cudaMalloc((void**)&dev_uint16, sizeof(uint16_t));
    cudaMemcpy(dev_uint16, &host_uint16, sizeof(uint16_t), cudaMemcpyHostToDevice);

    uint32_t host_uint32 = 32;
    uint32_t *dev_uint32;
    cudaMalloc((void**)&dev_uint32, sizeof(uint32_t));
    cudaMemcpy(dev_uint32, &host_uint32, sizeof(uint32_t), cudaMemcpyHostToDevice);

    uint64_t host_uint64 = 64;
    uint64_t *dev_uint64;
    cudaMalloc((void**)&dev_uint64, sizeof(uint64_t));
    cudaMemcpy(dev_uint64, &host_uint64, sizeof(uint64_t), cudaMemcpyHostToDevice);

    test_ld_funcs<<<1, 1>>>(dev_int, dev_uint8, dev_uint16, dev_uint32, dev_uint64);
    cudaDeviceSynchronize();

    cudaFree(dev_int);
    cudaFree(dev_uint8);
    cudaFree(dev_uint16);
    cudaFree(dev_uint32);
    cudaFree(dev_uint64);
    return 0;
}
