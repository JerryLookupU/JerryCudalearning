#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <iostream>
#include <chrono>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

__global__ void elementwise_add_f32_kernel(float* a,float* b, float* c,int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        c[idx] = a[idx] + b[idx];
    }
}

// ElementWise Add + Vec4
// grid(N/256), block(256/4)
// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f32x4_kernel(float* a,float* b, float* c,int N){
   int idx = 4*(blockIdx.x * blockDim.x + threadIdx.x);
    if(idx < N){
        float4 reg_a = FLOAT4(a[idx]);
        float4 reg_b = FLOAT4(b[idx]);
        float4 reg_c;
        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        reg_c.z = reg_a.z + reg_b.z;
        reg_c.w = reg_a.w + reg_b.w;
        FLOAT4(c[idx]) = reg_c;
    } 
}

__global__ void elementwise_add_f16x2_kernel(half* a,half* b, half* c,int N){
    int idx = 2*(blockIdx.x * blockDim.x + threadIdx.x);
    if(idx < N){
        half2 reg_a = HALF2(a[idx]);
        half2 reg_b = HALF2(b[idx]);
        half2 reg_c;
        reg_c.x = __hadd(reg_a.x, reg_b.x);
        reg_c.y = __hadd(reg_a.y, reg_b.y);
        HALF2(c[idx]) = reg_c; 
    } 
}

__global__ void elementwise_add_f16x8_kernel(half* a,half* b, half* c,int N){
    int idx = 8*(blockIdx.x * blockDim.x + threadIdx.x);
    half2 reg_a_0 = HALF2(a[idx + 0]);
    half2 reg_a_1 = HALF2(a[idx + 2]);
    half2 reg_a_2 = HALF2(a[idx + 4]);
    half2 reg_a_3 = HALF2(a[idx + 6]);

    half2 reg_b_0 = HALF2(b[idx + 0]);
    half2 reg_b_1 = HALF2(b[idx + 2]);
    half2 reg_b_2 = HALF2(b[idx + 4]);
    half2 reg_b_3 = HALF2(b[idx + 6]);
    
    half2 reg_c_0,reg_c_1,reg_c_2,reg_c_3;
    reg_c_0.x = __hadd(reg_a_0.x, reg_b_0.x);
    reg_c_0.y = __hadd(reg_a_0.y, reg_b_0.y);
    reg_c_1.x = __hadd(reg_a_1.x, reg_b_1.x);
    reg_c_1.y = __hadd(reg_a_1.y, reg_b_1.y);
    reg_c_2.x = __hadd(reg_a_2.x, reg_b_2.x);
    reg_c_2.y = __hadd(reg_a_2.y, reg_b_2.y);
    reg_c_3.x = __hadd(reg_a_3.x, reg_b_3.x);
    reg_c_3.y = __hadd(reg_a_3.y, reg_b_3.y);

    if (idx < N) {
        HALF2(c[idx + 0]) = reg_c_0;
    }
    if ((idx + 2) < N) {
        HALF2(c[idx + 2]) = reg_c_1;
    }
    if ((idx + 4) < N) {
        HALF2(c[idx + 4]) = reg_c_2;
   
    }
    if ((idx + 6) < N) {
        HALF2(c[idx + 6]) = reg_c_3; 
    }

}


int main() {
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);
    size_t size_half = N * sizeof(half); // 1M elements, each element is 16b
    // Allocate host memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    
    half *half_a = (half*)malloc(size_half);
    half *half_b = (half*)malloc(size_half);
    half *half_c = (half*)malloc(size_half);


    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    for (int i = 0; i < N; i++) {
        half_a[i] = half(1.0f); // Convert float to hal
        half_b[i] = half(2.0f);
    }

    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    half *d_half_a, *d_half_b, *d_half_c;
    cudaMalloc(&d_half_a, size_half);
    cudaMalloc(&d_half_b, size_half);
    cudaMalloc(&d_half_c, size_half);

    
    // Copy data to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // copy half data to device
    cudaMemcpy(d_half_a, half_a, size_half, cudaMemcpyHostToDevice);
    cudaMemcpy(d_half_b, half_b, size_half, cudaMemcpyHostToDevice);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Test f32 kernel
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    cudaEventRecord(start);
    elementwise_add_f32_kernel<<<grid, block>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "f32 kernel time: " << milliseconds << " ms" << std::endl;
    
    // Test f32x4 kernel
    dim3 block4(64);  // 256 threads / 4 elements per thread
    dim3 grid4((N + block4.x * 4 - 1) / (block4.x * 4));
    
    cudaEventRecord(start);
    elementwise_add_f32x4_kernel<<<grid4, block4>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "f32x4 kernel time: " << milliseconds << " ms" << std::endl;
    

    // Test f16x2 kernel
    dim3 block2(128);  // 256 threads / 2 elements per thread
    dim3 grid2((N + block2.x * 2 - 1) / (block2.x * 2) );
    cudaEventRecord(start);
    elementwise_add_f16x2_kernel<<<grid2, block2>>>(d_half_a, d_half_b, d_half_c, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "f16x2 kernel time: " << milliseconds << " ms" << std::endl;
    
    // Test f16x8 kernel
    dim3 block8(32);  // 256 threads / 8 elements per thread
    dim3 grid8((N + block8.x * 8 - 1) / (block8.x * 8));
    
    cudaEventRecord(start);
    elementwise_add_f16x8_kernel<<<grid8, block8>>>(d_half_a, d_half_b, d_half_c, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "f16x8 kernel time: " << milliseconds << " ms" << std::endl;

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    cudaFree(d_half_a);
    cudaFree(d_half_b);
    cudaFree(d_half_c);
    free(half_a);
    free(half_b);
    free(half_c);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}