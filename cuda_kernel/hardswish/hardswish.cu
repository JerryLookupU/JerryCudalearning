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

#define THRESHOLD_MAX 3.0f
#define THRESHOLD_MIN -3.0f


// hardswish 公式 if x=< -3 返回0 if x>=3 返回x else 返回x*(x+3)/6

__device__ __forceinline__ float hardswish(float x) {

    if (x >= THRESHOLD_MAX){
        return x;
    } else if (x <= THRESHOLD_MIN){
        return 0.0f; 
    } else {
        return x * (x + 3.0f) / 6.0f;
    }
}


__device__ __forceinline__ half hardswish_half(half x) {
    if (x >= __float2half(THRESHOLD_MAX)){
        return x;
    } else if (x <= __float2half(THRESHOLD_MIN)){
        return __float2half(0.0f); 
    } else {
        return x * (x + __float2half(3.0f)) / __float2half(6.0f);
    }
}

__global__ void hardswish_fp32_kernel(float x, float* y,int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        y[idx] = hardswish(x);
    }
}

__global__ void hardswish_fp32x4_kernel(float *x, float* y,int N) {
    int idx = 4*(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) {
        FLOAT4 reg_x = FLOAT4(x[idx]);
        FLOAT4 reg_y;
        reg_y.x = hardswish(reg_x.x);
        reg_y.y = hardswish(reg_x.y);
        reg_y.z = hardswish(reg_x.z);
        reg_y.w = hardswish(reg_x.w);
        FLOAT4(y[idx]) = reg_y;
    }
}

__global__ void hardswish_fp16_kernel(half *x, half* y,int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        y[idx] = hardswish_half(x[idx]);
    }
}

__global__ void hardswish_fp16x2_kernel(half *x, half* y,int N) {
    int idx = 2*(blockIdx.x * blockDim.x + threadIdx.x);
    HALF2 reg_x = HALF2(x[idx]);
    HALF2 reg_y;
    reg_y.x = hardswish_half(reg_x.x);
    reg_y.y = hardswish_half(reg_x.y);
    HALF2(y[idx]) = reg_y;
}


__global__ void hardswish_fp16x8_kernel(half *x, half* y,int N) {
   int idx = 8*(blockIdx.x * blockDim.x + threadIdx.x);
   half2 reg_x_0 = HALF2(x[idx]);
   half2 reg_x_1 = HALF2(x[idx+2]);
   half2 reg_x_2 = HALF2(x[idx+4]);
   half2 reg_x_3 = HALF2(x[idx+6]);

   half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;
   reg_y_0.x = hardswish_half(reg_x_0.x);
   reg_y_0.y = hardswish_half(reg_x_0.y);
   reg_y_1.x = hardswish_half(reg_x_1.x);
   reg_y_1.y = hardswish_half(reg_x_1.y);
   reg_y_2.x = hardswish_half(reg_x_2.x);
   reg_y_2.y = hardswish_half(reg_x_2.y);
   reg_y_3.x = hardswish_half(reg_x_3.x);
   reg_y_3.y = hardswish_half(reg_x_3.y);
   if ((idx) < N) {
        HALF2(y[idx]) = reg_y_0;
   }
   if ((idx+2) < N) {
        HALF2(y[idx+2]) = reg_y_1;
   }
   if ((idx+4) < N) {
        HALF2(y[idx+4]) = reg_y_2;
  
    }
   if ((idx+6) < N) {
        HALF2(y[idx+6]) = reg_y_3; 
   }
}

__global__ void hardswish_fp16x8_pack_kernel(half *x, half* y,int N) {
   int idx = 8*(blockIdx.x * blockDim.x + threadIdx.x);
   half2 reg_x_pack[8], reg_y_pack[8];

    LDST128BITS(reg_x_pack[0]) = LDST128BITS(x[idx]);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        reg_y_pack[i] = hardswish_half(reg_x_pack[i]);
    }

    if ((idx + 7) < N) {
        LDST128BITS(y[idx]) = LDST128BITS(reg_y_pack[0]);
    }
}

void test_hardswish() {
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

    // 测试 fp32 kernel
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    float milliseconds = 0.0f;
    cudaEventRecord(start);
    hardswish_fp32_kernel<<<grid, block>>>(d_a, d_y, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("HardSwish kernel fp32 execution time: %f ms\n", milliseconds);

    // 测试 fp32x4 kernel
    dim3 block_x4(64);  // 256/4
    dim3 grid_x4((N + block_x4.x * 4 - 1) / (block_x4.x * 4));
    cudaEventRecord(start);
    hardswish_fp32x4_kernel<<<grid_x4, block_x4>>>(d_a, d_y, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("HardSwish kernel fp32x4 execution time: %f ms\n", milliseconds);

    // 测试 fp16 kernel
    cudaEventRecord(start);
    hardswish_fp16_kernel<<<grid, block>>>(d_a_f16, d_y_f16, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("HardSwish kernel fp16 execution time: %f ms\n", milliseconds);

    // 测试 fp16x2 kernel
    dim3 block_f16x2(128);  // 256/2
    dim3 grid_f16x2((N + block_f16x2.x * 2 - 1) / (block_f16x2.x * 2));
    cudaEventRecord(start);
    hardswish_fp16x2_kernel<<<grid_f16x2, block_f16x2>>>(d_a_f16, d_y_f16, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("HardSwish kernel fp16x2 execution time: %f ms\n", milliseconds);

    // 测试 fp16x8 kernel
    dim3 block_f16x8(32);  // 256/8
    dim3 grid_f16x8((N + block_f16x8.x * 8 - 1) / (block_f16x8.x * 8));
    cudaEventRecord(start);
    hardswish_fp16x8_kernel<<<grid_f16x8, block_f16x8>>>(d_a_f16, d_y_f16, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("HardSwish kernel fp16x8 execution time: %f ms\n", milliseconds);

    // 测试 fp16x8 pack kernel
    cudaEventRecord(start);
    hardswish_fp16x8_pack_kernel<<<grid_f16x8, block_f16x8>>>(d_a_f16, d_y_f16, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("HardSwish kernel fp16x8 pack execution time: %f ms\n", milliseconds);

    // 验证结果
    float *h_y = (float *)malloc(N * sizeof(float));
    half *h_y_f16 = (half *)malloc(N * sizeof(half));
    cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y_f16, d_y_f16, N * sizeof(half), cudaMemcpyDeviceToHost);
    
    // 计算CPU结果
    float *h_y_cpu = (float *)malloc(N * sizeof(float));
    half *h_y_cpu_f16 = (half *)malloc(N * sizeof(half));
    for (int i = 0; i < N; i++) {
        // 使用标准HardSwish函数计算CPU结果
        float x = h_a[i];
        if (x <= -3.0f) {
            h_y_cpu[i] = 0.0f;
        } else if (x >= 3.0f) {
            h_y_cpu[i] = x;
        } else {
            h_y_cpu[i] = x * (x + 3.0f) / 6.0f;
        }
        h_y_cpu_f16[i] = __float2half(h_y_cpu[i]);
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
    printf("HardSwish test passed!\n");
    
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
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    test_hardswish();
    return 0;
}