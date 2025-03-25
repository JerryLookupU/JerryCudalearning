#include <stdio.h>
__device__ __forceinline__ int atomic_add_release_sys_global(const int* ptr, int value) {
    int ret;
    asm volatile("atom.add.release.sys.global.s32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(value));
    return ret;
}

__device__ __forceinline__ int atomic_add_release_global(const int* ptr, int value) {
    int ret;
    asm volatile("atom.add.release.gpu.global.s32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(value));
    return ret;
}


__global__ void test_atomic_add_release_sys_global(int* result) {
    atomic_add_release_sys_global(result, 1);
}

__global__ void test_atomic_add_release_global(int* result) {
    atomic_add_release_global(result, 1);
}

int main() {
    int* d_result;
    cudaMalloc(&d_result, sizeof(int));
    cudaMemset(d_result, 0, sizeof(int));

    // 测试sys版本
    test_atomic_add_release_sys_global<<<1, 10>>>(d_result);
    cudaDeviceSynchronize();

    // 测试gpu版本
    test_atomic_add_release_global<<<1, 10>>>(d_result);
    cudaDeviceSynchronize();

    int h_result;
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Final result: %d\n", h_result);  // 预期输出20 (10线程×2次调用)

    cudaFree(d_result);
    return 0;
}