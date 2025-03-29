#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__forceinline__ __device__ int warp_reduce_sum(int value) {
    value += __shfl_xor_sync(0xffffffff, value, 16);
    value += __shfl_xor_sync(0xffffffff, value, 8);
    value += __shfl_xor_sync(0xffffffff, value, 4);
    value += __shfl_xor_sync(0xffffffff, value, 2);
    value += __shfl_xor_sync(0xffffffff, value, 1);
    return value;
}

__forceinline__ __device__ float half_warp_reduce_max(float value) {
    auto mask = __activemask();
    // The mask be in `{0xffffffff, 0xffff}`
    value = max(value, __shfl_xor_sync(mask, value, 8));
    value = max(value, __shfl_xor_sync(mask, value, 4));
    value = max(value, __shfl_xor_sync(mask, value, 2));
    value = max(value, __shfl_xor_sync(mask, value, 1));
    return value;
}

__global__ void test_warp_reduce_sum(int* input, int* output) {
    int tid = threadIdx.x;
    output[tid] = warp_reduce_sum(input[tid]);
    // 一个warp 通常32个线程
}

__global__ void test_half_warp_reduce_max(float* input, float* output) {
    int tid = threadIdx.x;
    output[tid] = half_warp_reduce_max(input[tid]);
}

void run_tests() {
    const int size = 32;
    int* h_input = new int[size];
    int* h_output = new int[size];
    float* h_finput = new float[size];
    float* h_foutput = new float[size];
    
    // 测试warp_reduce_sum
    for (int i = 0; i < size; ++i) h_input[i] = i + 1;
    
    int *d_input, *d_output;
    float *d_finput, *d_foutput;
    cudaMalloc(&d_input, size * sizeof(int));
    cudaMalloc(&d_output, size * sizeof(int));
    cudaMemcpy(d_input, h_input, size * sizeof(int), cudaMemcpyHostToDevice);
    
    test_warp_reduce_sum<<<1, size>>>(d_input, d_output);
    cudaMemcpy(h_output, d_output, size * sizeof(int), cudaMemcpyDeviceToHost);
    
    // 验证结果
    int expected_sum = (size * (size + 1)) / 2;
    for (int i = 0; i < size; ++i) {
        if (h_output[i] != expected_sum) {
            printf("warp_reduce_sum test failed at thread %d: got %d, expected %d\n", 
                   i, h_output[i], expected_sum);
        }
    }
    
    // 测试half_warp_reduce_max
    for (int i = 0; i < size; ++i) h_finput[i] = (float)(i + 1);
    
    cudaMalloc(&d_finput, size * sizeof(float));
    cudaMalloc(&d_foutput, size * sizeof(float));
    cudaMemcpy(d_finput, h_finput, size * sizeof(float), cudaMemcpyHostToDevice);
    
    test_half_warp_reduce_max<<<1, size>>>(d_finput, d_foutput);
    cudaMemcpy(h_foutput, d_foutput, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 验证结果 - only check first 16 threads since it's a half-warp operation
    float expected_max = (float)16;
    for (int i = 0; i < 16; ++i) {
        if (h_foutput[i] != expected_max) {
            printf("half_warp_reduce_max test failed at thread %d: got %f, expected %f\n", 
                   i, h_foutput[i], expected_max);
        }
    }
    
    // 清理
    delete[] h_input;
    delete[] h_output;
    delete[] h_finput;
    delete[] h_foutput;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_finput);
    cudaFree(d_foutput);
}

int main() {
    run_tests();
    printf("All tests completed.\n");
    return 0;
}
