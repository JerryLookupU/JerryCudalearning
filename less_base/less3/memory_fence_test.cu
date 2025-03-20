#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
__device__ __forceinline__ void memory_fence() {
    asm volatile("fence.acq_rel.sys;":: : "memory");
}

__device__ __forceinline__ void memory_fence_gpu() {
    asm volatile("fence.acq_rel.gpu;":: : "memory");
}

__device__ __forceinline__ void memory_fence_cta() {
    asm volatile("fence.acq_rel.cta;":: : "memory");
}

// 场景1：多CTA共享数据时使用sys级屏障 # 加了 memory_fence 后，global_data 值最终返回值
__global__ void test_sys_level(int* global_counter) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int cta = tid + bid * blockDim.x;
    int old_val = atomicAdd(global_counter, cta);
    printf("[Before sys fence] Block %d Thread %d - Old value: %d\n", bid, tid, old_val);
    printf("[AtomicAdd] Block %d Thread %d - New value: %d\n", bid, tid, old_val + cta);
    
    memory_fence(); // GPU级内存屏障
    
    printf("[After  sys fence] Block %d Thread %d - Final value: %d\n", bid, tid, *global_counter);
    memory_fence(); // sys级内存屏障

    printf("test_cta_level Block %d Thread %d added %d | cta: %d,Current Value %d\n", bid, tid, old_val,cta, *global_counter); 
    
    // printf("test_cta_level Block %d Thread %d added %d | cta: %d ,Current Value %d\n", bid, tid, old_val,cta , *global_counter);
    __syncthreads();
}

// 场景2：块内线程协作时用cta级屏障 memory_fence_cta  global_counter 分成 两次输出（中间一次 和 最后一次）
__global__ void test_cta_level(int* global_counter) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int cta = tid + bid * blockDim.x;
    int old_val = atomicAdd(global_counter, cta);
    printf("[Before cta fence] Block %d Thread %d - Old value: %d\n", bid, tid, old_val);
    printf("[AtomicAdd] Block %d Thread %d - New value: %d\n", bid, tid, old_val + cta);
    
    memory_fence_cta(); // GPU级内存屏障
    
    printf("[After cta fence] Block %d Thread %d - Final value: %d\n", bid, tid, *global_counter);
    memory_fence_cta(); // sys级内存屏障

    printf("test_cta_level Block %d Thread %d added %d | cta: %d,Current Value %d\n", bid, tid, old_val,cta, *global_counter); 
    
    // printf("test_cta_level Block %d Thread %d added %d | cta: %d ,Current Value %d\n", bid, tid, old_val,cta , *global_counter);
    __syncthreads();
}

// 场景3：GPU全局数据同步时用gpu级屏障
__global__ void test_gpu_level(int* global_counter) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int cta = tid + bid * blockDim.x;
    int old_val = atomicAdd(global_counter, cta);
    printf("[Before fence] Block %d Thread %d - Old value: %d\n", bid, tid, old_val);
    printf("[AtomicAdd] Block %d Thread %d - New value: %d\n", bid, tid, old_val + cta);
    
    memory_fence_gpu(); // GPU级内存屏障
    
    printf("[After fence] Block %d Thread %d - Final value: %d\n", bid, tid, *global_counter);
    memory_fence_gpu(); // sys级内存屏障

    printf("test_gpu_level Block %d Thread %d added %d | cta: %d,Current Value %d\n", bid, tid, old_val,cta, *global_counter); 
    
    // printf("test_cta_level Block %d Thread %d added %d | cta: %d ,Current Value %d\n", bid, tid, old_val,cta , *global_counter);
    __syncthreads();
}

__global__ void test_sys_level(int* sys_counter);
__global__ void test_cta_level(int* cta_counter);
__global__ void test_gpu_level(int* gpu_counter);

void print_results(int sys_val, int cta_val, int gpu_val) {
    printf("Sys-level (2 blocks * 4 threads): %d\n", sys_val);
    printf("CTA-level (2 blocks * 4 threads): %d\n", cta_val);
    printf("GPU-level (2 blocks * 4 threads): %d\n", gpu_val);
}

int main() {
    int *d_sys_counter, *d_cta_counter, *d_gpu_counter;
    int h_sys_counter = 0, h_cta_counter = 0, h_gpu_counter = 0;

    cudaMalloc(&d_sys_counter, sizeof(int));
    cudaMalloc(&d_cta_counter, sizeof(int));
    cudaMalloc(&d_gpu_counter, sizeof(int));
    cudaMemset(d_sys_counter, 0, sizeof(int));
    cudaMemset(d_cta_counter, 0, sizeof(int));
    cudaMemset(d_gpu_counter, 0, sizeof(int));

    test_sys_level<<<2, 4, 16*sizeof(int)>>>(d_sys_counter);
    test_cta_level<<<2, 4>>>(d_cta_counter);
    test_gpu_level<<<2, 4>>>(d_gpu_counter);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_sys_counter, d_sys_counter, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_cta_counter, d_cta_counter, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_gpu_counter, d_gpu_counter, sizeof(int), cudaMemcpyDeviceToHost);

    print_results(h_sys_counter, h_cta_counter, h_gpu_counter);

    cudaFree(d_sys_counter);
    cudaFree(d_cta_counter);
    cudaFree(d_gpu_counter);
    return 0;
}