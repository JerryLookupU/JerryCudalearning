#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])


// -------------------------------------- FP32 -------------------------------------- 
// Relu x: N, y: N y=max(0,x)
// grid(N/256), block(K=256) 
__global__ void relu_f32_kernel(float* x, float* y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) y[idx] = fmaxf(0.0f, x[idx]);
  }
  
  // Relu x: N, y: N y=max(0,x) Vec4
  // grid(N/256/4), block(256/4) 
  __global__ void relu_f32x4_kernel(float* x, float* y, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx < N) {
      float4 reg_x = FLOAT4(x[idx]);
      float4 reg_y;
      reg_y.x = fmaxf(0.0f, reg_x.x);
      reg_y.y = fmaxf(0.0f, reg_x.y);
      reg_y.z = fmaxf(0.0f, reg_x.z);
      reg_y.w = fmaxf(0.0f, reg_x.w);
      FLOAT4(y[idx]) = reg_y;
    }
  }
  
  // -------------------------------------- FP16 -------------------------------------- 
  __global__ void relu_f16_kernel(half* x, half* y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) y[idx] = __hmax(__float2half(0.0f), x[idx]);
  }
  
  __global__ void relu_f16x2_kernel(half* x, half* y, int N) {
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) {
      half2 reg_x = HALF2(x[idx]);
      half2 reg_y = HALF2(y[idx]);
      reg_y.x = __hmax(__float2half(0.0f), reg_x.x);
      reg_y.y = __hmax(__float2half(0.0f), reg_x.y);
      HALF2(y[idx]) = reg_y;
    }
  }
  
  __global__ void relu_f16x8_kernel(half* x, half* y, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    half2 reg_x_0 = HALF2(x[idx + 0]);
    half2 reg_x_1 = HALF2(x[idx + 2]);
    half2 reg_x_2 = HALF2(x[idx + 4]);
    half2 reg_x_3 = HALF2(x[idx + 6]);
    half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;
    reg_y_0.x = __hmax(__float2half(0.0f), reg_x_0.x);
    reg_y_0.y = __hmax(__float2half(0.0f), reg_x_0.y);
    reg_y_1.x = __hmax(__float2half(0.0f), reg_x_1.x);
    reg_y_1.y = __hmax(__float2half(0.0f), reg_x_1.y);
    reg_y_2.x = __hmax(__float2half(0.0f), reg_x_2.x);
    reg_y_2.y = __hmax(__float2half(0.0f), reg_x_2.y);
    reg_y_3.x = __hmax(__float2half(0.0f), reg_x_3.x);
    reg_y_3.y = __hmax(__float2half(0.0f), reg_x_3.y);
    if ((idx + 0) < N) { HALF2(y[idx + 0]) = reg_y_0; }
    if ((idx + 2) < N) { HALF2(y[idx + 2]) = reg_y_1; }
    if ((idx + 4) < N) { HALF2(y[idx + 4]) = reg_y_2; }
    if ((idx + 6) < N) { HALF2(y[idx + 6]) = reg_y_3; }
  }
  
  __global__ void relu_f16x8_pack_kernel(half* x, half* y, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    const half2 z2 = {__float2half(0.0f), __float2half(0.0f)};
    // temporary register(memory), .local space in ptx, addressable
    half pack_x[8], pack_y[8]; // 8x16 bits=128 bits.
    // reinterpret as float4 and load 128 bits in 1 memory issue.
    LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]); // load 128 bits
  
    #pragma unroll
    for (int i = 0; i < 8; i += 2) {
      // __hmax2 for half2 x 4
      HALF2(pack_y[i]) = __hmax2(HALF2(pack_x[i]), z2);
    } 
    // reinterpret as float4 and store 128 bits in 1 memory issue.
    if ((idx + 7) < N) { LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]); }
  }
  // -------------------------------------- BF16 --------------------------------------  



void test_relu(){
    const int N = 1024;
    
    // 生成测试数据
    float *h_a = (float *)malloc(N * sizeof(float));
    half *h_a_f16 = (half *)malloc(N * sizeof(half));
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)rand() / RAND_MAX * 10.0f - 5.0f; // 生成-5到5之间的随机数
        h_a_f16[i] = __float2half(h_a[i]);
    }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 分配设备内存
    float *d_a, *d_y;
    half *d_a_f16, *d_y_f16;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_a_f16, N * sizeof(half));
    cudaMalloc(&d_y_f16, N * sizeof(half));
    
    // 拷贝数据到设备
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_f16, h_a_f16, N * sizeof(half), cudaMemcpyHostToDevice);

    // 测试 f32 kernel
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    float milliseconds = 0.0f;
    cudaEventRecord(start);
    relu_f32_kernel<<<grid, block>>>(d_a, d_y, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("ReLU kernel f32 execution time: %f ms\n", milliseconds);

    // 测试 f32x4 kernel
    dim3 block_x4(64);  // 256/4
    dim3 grid_x4((N + block_x4.x * 4 - 1) / (block_x4.x * 4));
    cudaEventRecord(start);
    relu_f32x4_kernel<<<grid_x4, block_x4>>>(d_a, d_y, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("ReLU kernel f32x4 execution time: %f ms\n", milliseconds);

    // 测试 f16 kernel
    cudaEventRecord(start);
    relu_f16_kernel<<<grid, block>>>(d_a_f16, d_y_f16, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("ReLU kernel f16 execution time: %f ms\n", milliseconds);

    // 测试 f16x2 kernel
    dim3 block_f16x2(128);  // 256/2
    dim3 grid_f16x2((N + block_f16x2.x * 2 - 1) / (block_f16x2.x * 2));
    cudaEventRecord(start);
    relu_f16x2_kernel<<<grid_f16x2, block_f16x2>>>(d_a_f16, d_y_f16, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("ReLU kernel f16x2 execution time: %f ms\n", milliseconds);

    // 验证结果
    float *h_y = (float *)malloc(N * sizeof(float));
    half *h_y_f16 = (half *)malloc(N * sizeof(half));
    cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y_f16, d_y_f16, N * sizeof(half), cudaMemcpyDeviceToHost);
    
    // 计算CPU结果
    float *h_y_cpu = (float *)malloc(N * sizeof(float));
    half *h_y_cpu_f16 = (half *)malloc(N * sizeof(half));
    for (int i = 0; i < N; i++) {
        h_y_cpu[i] = fmaxf(0.0f, h_a[i]);
        h_y_cpu_f16[i] = __hmax(__float2half(0.0f), h_a_f16[i]);
    }
    
    // 比较结果
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
    printf("ReLU test passed!\n");
    
    // 释放资源
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
}

int main(){
    test_relu();
    return 0; 
}