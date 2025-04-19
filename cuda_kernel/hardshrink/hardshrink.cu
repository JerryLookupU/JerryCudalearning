#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])
#define MAX_EXP_F32  88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f
#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)
#define HALF_1 __float2half(1.0f)

#define THRESHOLD_MAX 0.5f
#define THRESHOLD_MIN -0.5f
#define THRESHOLD_MAX_HALF __float2half(THRESHOLD_MAX)
#define THRESHOLD_MIN_HALF __float2half(THRESHOLD_MIN)

__device__ __forceinline__ float hardshrink(float x) {
    if (x >= THRESHOLD_MAX){
        return x;
    } else if (x <= THRESHOLD_MIN){
        return 0.0f;
    } else {
        return x;
    }
}

__device__ __forceinline__ half hardshrink_half(half x) {

    if (x >= THRESHOLD_MAX_HALF){
        return x;
    } else if (x <= THRESHOLD_MIN_HALF){
        return __float2half(0.0f);
    } else {
        return x;
    }
}

__global__ void hardshrink_fp32_kernel(float *input, float *output, int num_elements) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        output[tid] = hardshrink(input[tid]);;
    }
}

__global__ void hardshrink_fp16_kernel(half *input, half *output, int num_elements) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        output[tid] = hardshrink_half(input[tid]);
    }
}

__global__ void hardshrink_fp32x4_kernel(float *input, float *output, int N) {

    int tid = 4*( threadIdx.x + blockIdx.x * blockDim.x);
    if (tid < N) {
        FLOAT4 reg_x = FLOAT4(input[tid]);
        FLOAT4 reg_y;

        reg_y.x = hardshrink(reg_x.x);
        reg_y.y = hardshrink(reg_x.y);
        reg_y.z = hardshrink(reg_x.z);
        reg_y.w = hardshrink(reg_x.w);

        FLOAT4(output[tid]) = reg_y;
    }
}

__global__ void hardshrink_fp16x2_kernel(half *input, half *output, int N) {
    int tid = 2*( threadIdx.x + blockIdx.x * blockDim.x);
    if (tid < N) {
        HALF2 reg_x = HALF2(input[tid]);
        HALF2 reg_y;

        reg_y.x = hardshrink_half(reg_x.x);
        reg_y.y = hardshrink_half(reg_x.y);

        HALF2(output[tid]) = reg_y;
    }
}

__global__ void hardshrink_fp16x8_kernel(half *input, half *output, int N) {
    int tid = 8*( threadIdx.x + blockIdx.x * blockDim.x);

    HALF2 reg_x_0,reg_x_1,reg_x_2,reg_x_3;
    HALF2 reg_y_0,reg_y_1,reg_y_2,reg_y_3;
    reg_x_0 = HALF2(input[tid]);
    reg_x_1 = HALF2(input[tid+2]);
    reg_x_2 = HALF2(input[tid+4]);
    reg_x_3 = HALF2(input[tid+6]);

    reg_y_0.x = hardshrink_half(reg_x_0.x);
    reg_y_0.y = hardshrink_half(reg_x_0.y);
    reg_y_1.x = hardshrink_half(reg_x_1.x);
    reg_y_1.y = hardshrink_half(reg_x_1.y);
    reg_y_2.x = hardshrink_half(reg_x_2.x);
    reg_y_2.y = hardshrink_half(reg_x_2.y);
    reg_y_3.x = hardshrink_half(reg_x_3.x);
    reg_y_3.y = hardshrink_half(reg_x_3.y);
    if (tid < N) {
        HALF2(output[tid]) = reg_y_0;
    }
    if (tid+2 < N) {
        HALF2(output[tid+2]) = reg_y_1;
    }
    if (tid+4 < N) {
        HALF2(output[tid+4]) = reg_y_2;
    }
    if (tid+6 < N) {
        HALF2(output[tid+6]) = reg_y_3;
    }   
    
}

__global__ void hardshrink_fp16x8_pack_kernel(half *input, half *output, int N) {

    int tid = 8*( threadIdx.x + blockIdx.x * blockDim.x);
    half2 reg_pack_x[4], reg_pack_y[4];
    LDST128BITS(reg_pack_x[0]) = LDST128BITS(input[tid]);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        reg_pack_y[i].x = hardshrink_half(reg_pack_x[i].x);
        reg_pack_y[i].y = hardshrink_half(reg_pack_x[i].y);
    }
    if （(tid + 7） < N) {
        LDST128BITS(output[tid]) = LDST128BITS(reg_pack_y[0]);
    }

}

void test_hardshrink() {
    const int N = 1024;
    
    // Generate test data
    float *h_a = (float *)malloc(N * sizeof(float));
    half *h_a_f16 = (half *)malloc(N * sizeof(half));
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f; // Generate -1 to 1 random numbers
        h_a_f16[i] = __float2half(h_a[i]);
    }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Allocate device memory
    float *d_a, *d_y;
    half *d_a_f16, *d_y_f16;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_a_f16, N * sizeof(half));
    cudaMalloc(&d_y_f16, N * sizeof(half));
    
    // Copy data to device
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_f16, h_a_f16, N * sizeof(half), cudaMemcpyHostToDevice);

    // Test fp32 kernel
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    float milliseconds = 0.0f;
    cudaEventRecord(start);
    hardshrink_fp32_kernel<<<grid, block>>>(d_a, d_y, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Hardshrink kernel fp32 execution time: %f ms\n", milliseconds);

    // Test fp32x4 kernel
    dim3 block_x4(64);  // 256/4
    dim3 grid_x4((N + block_x4.x * 4 - 1) / (block_x4.x * 4));
    cudaEventRecord(start);
    hardshrink_fp32x4_kernel<<<grid_x4, block_x4>>>(d_a, d_y, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Hardshrink kernel fp32x4 execution time: %f ms\n", milliseconds);

    // Test fp16 kernel
    cudaEventRecord(start);
    hardshrink_fp16_kernel<<<grid, block>>>(d_a_f16, d_y_f16, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Hardshrink kernel fp16 execution time: %f ms\n", milliseconds);

    // Test fp16x2 kernel
    dim3 block_f16x2(128);  // 256/2
    dim3 grid_f16x2((N + block_f16x2.x * 2 - 1) / (block_f16x2.x * 2));
    cudaEventRecord(start);
    hardshrink_fp16x2_kernel<<<grid_f16x2, block_f16x2>>>(d_a_f16, d_y_f16, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Hardshrink kernel fp16x2 execution time: %f ms\n", milliseconds);

    // Test fp16x8 kernel
    dim3 block_f16x8(32);  // 256/8
    dim3 grid_f16x8((N + block_f16x8.x * 8 - 1) / (block_f16x8.x * 8));
    cudaEventRecord(start);
    hardshrink_fp16x8_kernel<<<grid_f16x8, block_f16x8>>>(d_a_f16, d_y_f16, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Hardshrink kernel fp16x8 execution time: %f ms\n", milliseconds);

    // Test fp16x8 pack kernel
    cudaEventRecord(start);
    hardshrink_fp16x8_pack_kernel<<<grid_f16x8, block_f16x8>>>(d_a_f16, d_y_f16, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Hardshrink kernel fp16x8 pack execution time: %f ms\n", milliseconds);

    // Verify results
    float *h_y = (float *)malloc(N * sizeof(float));
    half *h_y_f16 = (half *)malloc(N * sizeof(half));
    cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y_f16, d_y_f16, N * sizeof(half), cudaMemcpyDeviceToHost);
    
    // Calculate CPU results
    float *h_y_cpu = (float *)malloc(N * sizeof(float));
    half *h_y_cpu_f16 = (half *)malloc(N * sizeof(half));
    for (int i = 0; i < N; i++) {
        // Use standard Hardshrink function to calculate CPU results
        float x = h_a[i];
        h_y_cpu[i] = (x >= THRESHOLD_MAX) ? x : ((x <= THRESHOLD_MIN) ? 0.0f : x);
        h_y_cpu_f16[i] = __float2half(h_y_cpu[i]);
    }
    
    // Compare results
    float max_error = 0.0f;
    float max_error_f16 = 0.0f;
    for (int i = 0; i < N; i++) {
        float error = fabs(h_y[i] - h_y_cpu[i]);
        if (error > max_error) {
            max_error = error;
        }
        float error_f16 = fabs(__half2float(h_y_f16[i]) - __half2float(h_y_cpu_f16[i]));
        if (error_f16 > max_error_f16) {
            max_error_f16 = error_f16;
        }
    }
    printf("FP32 Max error: %f\n", max_error);
    printf("FP16 Max error: %f\n", max_error_f16);
    printf("Hardshrink test passed!\n");
    
    // Free resources
    free(h_a);
    free(h_a_f16);
    free(h_y);
    free(h_y_f16);
    free(h_y_cpu);
    free(h_y_cpu_f16);
    cudaFree(d_a);
    cudaFree(d_y);
    cudaFree(d_a_f16);
    cudaFree(d_y_f16);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    test_hardshrink();
    return 0;
}





