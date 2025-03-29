#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>


__device__ __forceinline__ int ld_acquire_sys_global(const int *ptr) {

    int ret;
    
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    
    return ret;
    
}
    
__device__ __forceinline__ uint64_t ld_acquire_sys_global(const uint64_t *ptr) {
    
    uint64_t ret;
    
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    
    return ret;
    
    }
    
__device__ __forceinline__ int ld_acquire_global(const int *ptr) {
    
    int ret;
    
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    
    return ret;
    
}
// 三个函数都是在CUDA里使用汇编指令直接进行内存加载操作的函数，它们各自有不同的用途和特点：
// 1.`ld_acquire_sys_global(const int *ptr)` ：从全局内存中读取一个32位整数，使用系统级的获取语义。这意味着在读取操作完成之前，后续的内存操作不会被执行，确保了内存访问的原子性和顺序性。
// 2.`ld_acquire_sys_global(const uint64_t *ptr)` ：功能与上面的函数类似，不过是读取一个64位无符号整数。
// 3.`ld_acquire_global(const int *ptr)` ：同样是从全局内存中读取一个32位整数，但使用的是GPU级别的获取语义，其作用范围主要在GPU内部，保证了在GPU内的内存操作顺序。    

__global__ void test_ld_acquire(int *dev_int, uint64_t *dev_uint64) {
    int result_int = ld_acquire_sys_global(dev_int);
    printf("ld_acquire_sys_global(int) result: %d\n", result_int);

    uint64_t result_uint64 = ld_acquire_sys_global(dev_uint64);
    printf("ld_acquire_sys_global(uint64_t) result: %llu\n", result_uint64);

    int result_global_int = ld_acquire_global(dev_int);
    printf("ld_acquire_global(int) result: %d\n", result_global_int);
}

int main() {
    int host_int = 42;
    int *dev_int;
    cudaMalloc((void**)&dev_int, sizeof(int));
    cudaMemcpy(dev_int, &host_int, sizeof(int), cudaMemcpyHostToDevice);

    uint64_t host_uint64 = 1234567890123456789ULL;
    uint64_t *dev_uint64;
    cudaMalloc((void**)&dev_uint64, sizeof(uint64_t));
    cudaMemcpy(dev_uint64, &host_uint64, sizeof(uint64_t), cudaMemcpyHostToDevice);

    test_ld_acquire<<<1, 1>>>(dev_int, dev_uint64);
    cudaDeviceSynchronize();

    cudaFree(dev_int);
    cudaFree(dev_uint64);
    return 0;
}