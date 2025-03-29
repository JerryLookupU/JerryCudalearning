#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#define NUM_BLOCKS 2
#define NUM_THREADS 4


__device__ __forceinline__ void st_relaxed_sys_global(int *ptr, int val) {
    asm volatile("st.relaxed.sys.global.s32 [%0], %1;"::"l"(ptr), "r"(val) : "memory");
}

__device__ __forceinline__ void st_release_sys_global(int *ptr, int val) {
    asm volatile("st.release.sys.global.s32 [%0], %1;"::"l"(ptr), "r"(val) : "memory");
}

__device__ __forceinline__ void st_release_cta(int *ptr, int val) {
    asm volatile("st.release.cta.s32 [%0], %1;"::"l"(ptr), "r"(val) : "memory");
}


__global__ void test_memory_store(int *global_ptr) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < NUM_BLOCKS * NUM_THREADS) {
        st_relaxed_sys_global(global_ptr + idx, idx);
        __syncthreads();
        st_release_sys_global(global_ptr + idx, idx * 2);
        __syncthreads();
        st_release_cta(global_ptr + idx, idx * 3);
    }
}

int main() {
    int *h_data = (int *)malloc(NUM_BLOCKS * NUM_THREADS * sizeof(int));
    int *d_data;
    cudaMalloc((void **)&d_data, NUM_BLOCKS * NUM_THREADS * sizeof(int));
    cudaMemset(d_data, 0, NUM_BLOCKS * NUM_THREADS * sizeof(int));

    dim3 dimBlock(NUM_THREADS);
    dim3 dimGrid(NUM_BLOCKS);
    test_memory_store<<<dimGrid, dimBlock>>>(d_data);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, NUM_BLOCKS * NUM_THREADS * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < NUM_BLOCKS * NUM_THREADS; i++) {
        printf("Data at index %d: %d\n", i, h_data[i]);
    }

    free(h_data);
    cudaFree(d_data);

    return 0;
}
