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


// MAX_EXP_F32 = 88.3762626647949f; // 2^15 * log(2) * (2 - 2^-23) 
// MIN_EXP_F32 = -88.3762626647949f; // -2^15 * log(2) * (2 - 2^-23)

__global__ void sigmoid_fp32_kernel(float* x,float* y, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // y[idx] = 1.0f / (1.0f + expf(-x[idx]));
        float v = x[idx];
        v = fminf(fmaxf(v, MIN_EXP_F32), MAX_EXP_F32);
        y[idx] = 1.0f / (1.0f + expf(-v));
    }
}


__global__ void sigmoid_fp32x4_kernel(float* x,float* y, int N){
    int idx = 4*(blockIdx.x * blockDim.x + threadIdx.x);
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;
    reg_y.x = fminf(fmaxf(reg_x.x, MIN_EXP_F32), MAX_EXP_F32);
    reg_y.y = fminf(fmaxf(reg_x.y, MIN_EXP_F32), MAX_EXP_F32);
    reg_y.z = fminf(fmaxf(reg_x.z, MIN_EXP_F32), MAX_EXP_F32);
    reg_y.w = fminf(fmaxf(reg_x.w, MIN_EXP_F32), MAX_EXP_F32);


    reg_y.x = 1.0f / (1.0f + expf(-reg_y.x));
    reg_y.y = 1.0f / (1.0f + expf(-reg_y.y));
    reg_y.z = 1.0f / (1.0f + expf(-reg_y.z));
    reg_y.w = 1.0f / (1.0f + expf(-reg_y.w));
    if ((idx + 0) < N) {FLOAT4(y[idx + 0]) = reg_y;}
}

__global__ void sigmoid_fp16_kernel(half* x,half* y, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const half f = __float2half(1.0f);
    if (idx < N) {
        half v = x[idx];
        v = __hmin(__hmax(v,MAX_EXP_F16), MIN_EXP_F16); 
        y[idx] = f / (f + hexp(-v)); 
        
    } 

}


__global__ void sigmoid_fp16x2_kernel(half* x,half* y, int N){
    int idx = 2*(blockIdx.x * blockDim.x + threadIdx.x);
    const half f = __float2half(1.0f);
    half2 reg_x = HALF2(x[idx]);
    half2 reg_y;
    
    reg_y.x = __hmin(__hmax(reg_x.x,MAX_EXP_F16), MIN_EXP_F16);
    reg_y.y = __hmin(__hmax(reg_x.y,MAX_EXP_F16), MIN_EXP_F16);

    reg_y.x = f / (f + hexp(-reg_y.x));
    reg_y.y = f / (f + hexp(-reg_y.y));

    if ((idx + 0) < N) {HALF2(y[idx + 0]) = reg_y;}
}

__global__void sigmoid_fp16x8_kernel(half* x,half* y, int N){
    int idx = 8*(blockIdx.x * blockDim.x + threadIdx.x);
    const half f = __float2half(1.0f);
    half2 reg_x1 = HALF2(x[idx]); 
    half2 reg_x2 = HALF2(x[idx + 2]);
    half2 reg_x3 = HALF2(x[idx + 4]);
    half2 reg_x4 = HALF2(x[idx + 6]);

    half2 reg_y1;
    half2 reg_y2;
    half2 reg_y3;
    half2 reg_y4;

    reg_y1.x = __hmin(__hmax(reg_x1.x,MAX_EXP_F16), MIN_EXP_F16);
    reg_y1.y = __hmin(__hmax(reg_x1.y,MAX_EXP_F16), MIN_EXP_F16);
    reg_y2.x = __hmin(__hmax(reg_x2.x,MAX_EXP_F16), MIN_EXP_F16);
    reg_y2.y = __hmin(__hmax(reg_x2.y,MAX_EXP_F16), MIN_EXP_F16);
    reg_y3.x = __hmin(__hmax(reg_x3.x,MAX_EXP_F16), MIN_EXP_F16);
    reg_y3.y = __hmin(__hmax(reg_x3.y,MAX_EXP_F16), MIN_EXP_F16);
    reg_y4.x = __hmin(__hmax(reg_x4.x,MAX_EXP_F16), MIN_EXP_F16);
    reg_y4.y = __hmin(__hmax(reg_x4.y,MAX_EXP_F16), MIN_EXP_F16);

    reg_y1.x = f / (f + hexp(-reg_y1.x));
    reg_y1.y = f / (f + hexp(-reg_y1.y));
    reg_y2.x = f / (f + hexp(-reg_y2.x));
    reg_y2.y = f / (f + hexp(-reg_y2.y));
    reg_y3.x = f / (f + hexp(-reg_y3.x));
    reg_y3.y = f / (f + hexp(-reg_y3.y));
    reg_y4.x = f / (f + hexp(-reg_y4.x));
    reg_y4.y = f / (f + hexp(-reg_y4.y));

    if ((idx + 0) < N) {HALF2(y[idx + 0]) = reg_y1; }
    if ((idx + 2) < N) {HALF2(y[idx + 2]) = reg_y2; }
    if ((idx + 4) < N) {HALF2(y[idx + 4]) = reg_y3; }
    if ((idx + 6) < N) {HALF2(y[idx + 6]) = reg_y4; }

}

__global__ void sigmoid_fp16x8_pack_kernel(half* x,half* y, int N){
    int idx = 8*(blockIdx.x * blockDim.x + threadIdx.x);
    const half f = __float2half(1.0f);
    half pack_x[8], pack_y[8];
    LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        half v = __hmin(__hmax(pack_x[i],MAX_EXP_F16), MIN_EXP_F16);
        pack_y[i] = f / (f + hexp(-v));
    }
    if ((idx + 7) < N) {
        LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
    }

}


void test_sigmoid() {
    const int N = 1024;
    
    // 生成测试数据
    float *h_a = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)rand() / RAND_MAX * 10.0f - 5.0f; // 生成-5到5之间的随机数
    }
    half *h_a_half = (half *)malloc(N * sizeof(half));
    for (int i = 0; i < N; i++) {
        h_a_half[i] = __float2half(h_a[i]); // 生成-5到5之间的随机数
    }
    half *h_b_half = (half *)malloc(N * sizeof(half));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 分配设备内存
    float *d_a, *d_y;
    half *d_a_half, *d_b_half;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_a_half, N * sizeof(half));
    cudaMalloc(&d_b_half, N * sizeof(half));
    
    // 拷贝数据到设备
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    // half
    cudaMemcpy(d_a_half, h_a_half, N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_half, h_b_half, N * sizeof(half), cudaMemcpyHostToDevice);


    // 调用核函数
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    float milliseconds = 0.0f;
    cudaEventRecord(start);
    sigmoid_fp32_kernel<<<grid, block>>>(d_a, d_y, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Sigmoid kernel execution time: %f ms\n", milliseconds);

    // 调用核函数

    milliseconds = 0.0f;
    cudaEventRecord(start);
    sigmoid_fp16_kernel<<<grid, block>>>(d_a_half, d_b_half, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Sigmoid fp16 kernel execution time: %f ms\n", milliseconds);


    // 调用核函数
    dim3 block4(64);
    dim3 grid4((N + block.x*4 - 1) / block.x*4);
    milliseconds = 0.0f;
    cudaEventRecord(start);
    sigmoid_fp32x4_kernel<<<grid4, block4>>>(d_a, d_y, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Sigmoid fp32x4 kernel execution time: %f ms\n", milliseconds);

    dim3 block2(128);
    dim3 grid2((N + block.x*2 - 1) / block.x*2);
    milliseconds = 0.0f;
    cudaEventRecord(start);
    sigmoid_fp16x2_kernel<<<grid2, block2>>>(d_a_half, d_b_half, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Sigmoid fp32x4 kernel execution time: %f ms\n", milliseconds);

    dim3 block8(64);
    dim3 grid8((N + block.x*8 - 1) / block.x*8);
    milliseconds = 0.0f;
    cudaEventRecord(start);
    sigmoid_fp16x8_kernel<<<grid8, block8>>>(d_a_half, d_b_half, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Sigmoid fp16x8 kernel execution time: %f ms\n", milliseconds);


    cudaEventRecord(start);
    sigmoid_fp16x8_pack_kernel<<<grid8, block8>>>(d_a_half, d_b_half, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Sigmoid fp16x8_pack kernel execution time: %f ms\n", milliseconds);
    // 验证结果
    float *h_y = (float *)malloc(N * sizeof(float));
    cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 计算CPU结果
    float *h_y_cpu = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        float v = h_a[i];
        v = fminf(fmaxf(v, MIN_EXP_F32), MAX_EXP_F32);
        h_y_cpu[i] = 1.0f / (1.0f + expf(-v));
    }
    
    // 比较结果
    float max_error = 0.0f;
    for (int i = 0; i < N; i++) {
        float error = fabs(h_y[i] - h_y_cpu[i]);
        if (error > max_error) {
            max_error = error;
        }
    }
    printf("Max error: %f\n", max_error);
    printf("Sigmoid test passed!\n");
    
    // 释放资源
    free(h_a);
    free(h_y);
    free(h_y_cpu);
    cudaFree(d_a);
    cudaFree(d_y);
}

int main() {
    test_sigmoid();
    return 0;
}
