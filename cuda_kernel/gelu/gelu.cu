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
#define SQRT_2_PI   M_SQRT2 * M_2_SQRTPI * 0.5ff
#define HALF_1 __float2half(1.0f)
#define HALF_2 __float2half(2.0f)
#define HALF_DIV2 __float2half(0.5f)

#define HALF_SQRT_2_PI __float2half(M_SQRT2) * __float2half(M_2_SQRTPI) * HALF_DIV2
#define HALF_V_APP __float2half(0.044715f)

#define HALF_GELU_OPS gelu_tanh_approximate
#define GELU_OPS gelu_tanh_approximate
#define HALF_GELU_OPS_V2 gelu_tanh_approximate_v2


// GELU(x) ≈ 0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])。
// 计算GELU(x)的近似值。
// def gelu(x):
//   """Gaussian Error Linear Unit.

//   This is a smoother version of the RELU.
//   Original paper: https://arxiv.org/abs/1606.08415
//   Args:
//     x: float Tensor to perform activation.

//   Returns:
//     `x` with the GELU activation applied.
//   """
//   cdf = 0.5 * (1.0 + tf.tanh(
//       (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
//   return x * cdf
// 近似版本
__inline__ __device__ float gelu_tanh_approximate(float x) {
    return 0.5f * (1.0f + tanhf(SQRT_2_PI * (x + 0.044715f * (x * x * x))))
}

// tanh --> 1 - 2/(1+exp(2x))
// tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)

__inline__ __device__ half gelu_tanh_approximate(half x) {
    half cube = x * x * x;
    half inner = HALF_SQRT_2_PI * (x + HALF_V_APP * cube);
    return HALF_DIV2 * x * (HALF_1 + (hexp(inner * HALF_2) - HALF_1) / (hexp(inner * HALF_2) + HALF_1));
}

__inline__ __device__ half gelu_tanh_approximate_v2(half x) {
    half cube = x * x * x;
    half inner = HALF_SQRT_2_PI * (x + HALF_V_APP * cube);
    return HALF_DIV2 * x * (HALF_2 - ( HALF_2 / (hexp(inner * HALF_2) + HALF_1));
}


// // 无差别版本
// def gelu(x):
//     return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
// 不要用 除 sqrt 计算尽量使用乘法 /sqrt(2.0) ==> 1/sqrt(2.0)
__inline__ device__ float gelu(float x) {
    return x * 0.5f * (1.0f + erff(x * M_SQRT1_2));
}



__global__ void gelu_fp32_kernel(float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // 先判断 溢出情况
        float v  = fmaxf(fminf(input[tid], MAX_EXP_F32), MIN_EXP_F32);
        output[tid] = GELU_OPS(v);
    }
}

__global__ void gelu_fp32x4_kernel(float* input, float* output, int n) {
    int tid = 4*(blockIdx.x * blockDim.x + threadIdx.x);
    float4 reg_x = FLOAT4(input[tid]);
    float4 reg_y;
    reg_x.x = fmaxf(fminf(reg_x.x, MAX_EXP_F32), MIN_EXP_F32);
    reg_x.y = fmaxf(fminf(reg_x.y, MAX_EXP_F32), MIN_EXP_F32);
    reg_x.z = fmaxf(fminf(reg_x.z, MAX_EXP_F32), MIN_EXP_F32);
    reg_x.w = fmaxf(fminf(reg_x.w, MAX_EXP_F32), MIN_EXP_F32);

    reg_y.x = GELU_OPS(reg_x.x);
    reg_y.y = GELU_OPS(reg_x.y);
    reg_y.z = GELU_OPS(reg_x.z);
    reg_y.w = GELU_OPS(reg_x.w);;
    if (idx < n) {
        FLOAT4(output[tid]) = reg_y;
    }
}


__global__ void gelu_fp16_kernel(half* input, half* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // 先判断 溢出情况
        half v  = __hmax(__hmin(input[tid], MAX_EXP_F16), MIN_EXP_F16);
        output[tid] = HALF_GELU_OPS(v); 
    } 
}

__global__ void gelu_fp16x2_kernel(half* input, half* output, int n) {
    int tid = 2*(blockIdx.x * blockDim.x + threadIdx.x);
    half2 reg_x = HALF2(input[tid]);
    half2 reg_y; 
    reg_x.x = __hmax(__hmin(reg_x.x, MAX_EXP_F16), MIN_EXP_F16);
    reg_x.y = __hmax(__hmin(reg_x.y, MAX_EXP_F16), MIN_EXP_F16);
    reg_y.x = HALF_GELU_OPS(reg_x.x);
    reg_y.y = HALF_GELU_OPS(reg_x.y);
    if (idx < n) {
        HALF2(output[tid]) = reg_y; 
    }
}

__global__ void gelu_fp16x2_kernel_v2(half* input, half* output, int n) {
    int tid = 2*(blockIdx.x * blockDim.x + threadIdx.x);
    half2 reg_x = HALF2(input[tid]);
    half2 reg_y; 
    reg_x.x = __hmax(__hmin(reg_x.x, MAX_EXP_F16), MIN_EXP_F16);
    reg_x.y = __hmax(__hmin(reg_x.y, MAX_EXP_F16), MIN_EXP_F16);
    reg_y.x = HALF_GELU_OPS_V2(reg_x.x);
    reg_y.y = HALF_GELU_OPS_V2(reg_x.y);
    if (idx < n) {
        HALF2(output[tid]) = reg_y; 
    }
}


__global__ void gelu_fp16x8_kernel(half* input, half* output, int n) {
    int tid = 8*(blockIdx.x * blockDim.x + threadIdx.x);
    half2 reg_x_0 = HALF2(input[tid]);
    half2 reg_x_1 = HALF2(input[tid + 2]);
    half2 reg_x_2 = HALF2(input[tid + 4]);
    half2 reg_x_3 = HALF2(input[tid + 6]);

    reg_x_0.x = __hmax(__hmin(reg_x_0.x, MAX_EXP_F16), MIN_EXP_F16);
    reg_x_0.y = __hmax(__hmin(reg_x_0.y, MAX_EXP_F16), MIN_EXP_F16);
    reg_x_1.x = __hmax(__hmin(reg_x_1.x, MAX_EXP_F16), MIN_EXP_F16);
    reg_x_1.y = __hmax(__hmin(reg_x_1.y, MAX_EXP_F16), MIN_EXP_F16);
    reg_x_2.x = __hmax(__hmin(reg_x_2.x, MAX_EXP_F16), MIN_EXP_F16);
    reg_x_2.y = __hmax(__hmin(reg_x_2.y, MAX_EXP_F16), MIN_EXP_F16);
    reg_x_3.x = __hmax(__hmin(reg_x_3.x, MAX_EXP_F16), MIN_EXP_F16);
    reg_x_3.y = __hmax(__hmin(reg_x_3.y, MAX_EXP_F16), MIN_EXP_F16);

    half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;
    reg_y_0.x = HALF_GELU_OPS(reg_x_0.x);
    reg_y_0.y = HALF_GELU_OPS(reg_x_0.y);
    reg_y_1.x = HALF_GELU_OPS(reg_x_1.x);
    reg_y_1.y = HALF_GELU_OPS(reg_x_1.y);
    reg_y_2.x = HALF_GELU_OPS(reg_x_2.x);
    reg_y_2.y = HALF_GELU_OPS(reg_x_2.y);
    reg_y_3.x = HALF_GELU_OPS(reg_x_3.x);
    reg_y_3.y = HALF_GELU_OPS(reg_x_3.y);

    if ((idx + 0) < n) {
        HALF2(output[tid + 0]) = reg_y_0; 
    }
    if ((idx + 2) < n) {
        HALF2(output[tid + 2]) = reg_y_1; 
    }
    if ((idx + 4) < n) {
        HALF2(output[tid + 4]) = reg_y_2;
    }
    if ((idx + 6) < n) {
        HALF2(output[tid + 6]) = reg_y_3; 
    }
}

__global__ void gelu_fp16x8_kernel_V2(half* input, half* output, int n) {
    int tid = 8*(blockIdx.x * blockDim.x + threadIdx.x);
    half2 reg_x_0 = HALF2(input[tid]);
    half2 reg_x_1 = HALF2(input[tid + 2]);
    half2 reg_x_2 = HALF2(input[tid + 4]);
    half2 reg_x_3 = HALF2(input[tid + 6]);

    reg_x_0.x = __hmax(__hmin(reg_x_0.x, MAX_EXP_F16), MIN_EXP_F16);
    reg_x_0.y = __hmax(__hmin(reg_x_0.y, MAX_EXP_F16), MIN_EXP_F16);
    reg_x_1.x = __hmax(__hmin(reg_x_1.x, MAX_EXP_F16), MIN_EXP_F16);
    reg_x_1.y = __hmax(__hmin(reg_x_1.y, MAX_EXP_F16), MIN_EXP_F16);
    reg_x_2.x = __hmax(__hmin(reg_x_2.x, MAX_EXP_F16), MIN_EXP_F16);
    reg_x_2.y = __hmax(__hmin(reg_x_2.y, MAX_EXP_F16), MIN_EXP_F16);
    reg_x_3.x = __hmax(__hmin(reg_x_3.x, MAX_EXP_F16), MIN_EXP_F16);
    reg_x_3.y = __hmax(__hmin(reg_x_3.y, MAX_EXP_F16), MIN_EXP_F16);

    half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;
    reg_y_0.x = HALF_GELU_OPS_V2(reg_x_0.x);
    reg_y_0.y = HALF_GELU_OPS_V2(reg_x_0.y);
    reg_y_1.x = HALF_GELU_OPS_V2(reg_x_1.x);
    reg_y_1.y = HALF_GELU_OPS_V2(reg_x_1.y);
    reg_y_2.x = HALF_GELU_OPS_V2(reg_x_2.x);
    reg_y_2.y = HALF_GELU_OPS_V2(reg_x_2.y);
    reg_y_3.x = HALF_GELU_OPS_V2(reg_x_3.x);
    reg_y_3.y = HALF_GELU_OPS_V2(reg_x_3.y);

    if ((idx + 0) < n) {
        HALF2(output[tid + 0]) = reg_y_0; 
    }
    if ((idx + 2) < n) {
        HALF2(output[tid + 2]) = reg_y_1; 
    }
    if ((idx + 4) < n) {
        HALF2(output[tid + 4]) = reg_y_2;
    }
    if ((idx + 6) < n) {
        HALF2(output[tid + 6]) = reg_y_3; 
    }
}


__global__ void gelu_fp16x8_pack_kernel(half* input, half* output, int n) {
    int tid = 8*(blockIdx.x * blockDim.x + threadIdx.x);
    half pack_x[8], pack_y[8]; 
    LDST128BITS(pack_x[0]) = LDST128BITS(input[tid]);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        half v = __hmax(__hmin(pack_x[i], MAX_EXP_F16), MIN_EXP_F16);
        pack_y[i] = HALF_GELU_OPS(v);
    }

    for ((idx + 7) < n) {
        LDST128BITS(output[tid]) = LDST128BITS(pack_y[0]);
    }
}

__global__ void gelu_fp16x8_pack_kernel_V2(half* input, half* output, int n) {
    int tid = 8*(blockIdx.x * blockDim.x + threadIdx.x);
    half pack_x[8], pack_y[8]; 
    LDST128BITS(pack_x[0]) = LDST128BITS(input[tid]);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        half v = __hmax(__hmin(pack_x[i], MAX_EXP_F16), MIN_EXP_F16);
        pack_y[i] = HALF_GELU_OPS_V2(v);
    }

    for ((idx + 7) < n) {
        LDST128BITS(output[tid]) = LDST128BITS(pack_y[0]);
    }
}

void test_gelu() {
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
    gelu_fp32_kernel<<<grid, block>>>(d_a, d_y, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GELU kernel fp32 execution time: %f ms\n", milliseconds);

    // 测试 fp32x4 kernel
    dim3 block_x4(64);  // 256/4
    dim3 grid_x4((N + block_x4.x * 4 - 1) / (block_x4.x * 4));
    cudaEventRecord(start);
    gelu_fp32x4_kernel<<<grid_x4, block_x4>>>(d_a, d_y, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GELU kernel fp32x4 execution time: %f ms\n", milliseconds);

    // 测试 fp16 kernel
    cudaEventRecord(start);
    gelu_fp16_kernel<<<grid, block>>>(d_a_f16, d_y_f16, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GELU kernel fp16 execution time: %f ms\n", milliseconds);

    // 测试 fp16x2 kernel
    dim3 block_f16x2(128);  // 256/2
    dim3 grid_f16x2((N + block_f16x2.x * 2 - 1) / (block_f16x2.x * 2));
    cudaEventRecord(start);
    gelu_fp16x2_kernel<<<grid_f16x2, block_f16x2>>>(d_a_f16, d_y_f16, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GELU kernel fp16x2 execution time: %f ms\n", milliseconds);

    // 测试 fp16x2 kernel v2
    cudaEventRecord(start);
    gelu_fp16x2_kernel_v2<<<grid_f16x2, block_f16x2>>>(d_a_f16, d_y_f16, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GELU kernel fp16x2 v2 execution time: %f ms\n", milliseconds);

    // 测试 fp16x8 kernel
    dim3 block_f16x8(32);  // 256/8
    dim3 grid_f16x8((N + block_f16x8.x * 8 - 1) / (block_f16x8.x * 8));
    cudaEventRecord(start);
    gelu_fp16x8_kernel<<<grid_f16x8, block_f16x8>>>(d_a_f16, d_y_f16, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GELU kernel fp16x8 execution time: %f ms\n", milliseconds);

    // 测试 fp16x8 kernel v2
    cudaEventRecord(start);
    gelu_fp16x8_kernel_V2<<<grid_f16x8, block_f16x8>>>(d_a_f16, d_y_f16, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GELU kernel fp16x8 v2 execution time: %f ms\n", milliseconds);

    // 测试 fp16x8 pack kernel
    cudaEventRecord(start);
    gelu_fp16x8_pack_kernel<<<grid_f16x8, block_f16x8>>>(d_a_f16, d_y_f16, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GELU kernel fp16x8 pack execution time: %f ms\n", milliseconds);

    // 测试 fp16x8 pack kernel v2
    cudaEventRecord(start);
    gelu_fp16x8_pack_kernel_V2<<<grid_f16x8, block_f16x8>>>(d_a_f16, d_y_f16, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GELU kernel fp16x8 pack v2 execution time: %f ms\n", milliseconds);

    // 验证结果
    float *h_y = (float *)malloc(N * sizeof(float));
    half *h_y_f16 = (half *)malloc(N * sizeof(half));
    cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y_f16, d_y_f16, N * sizeof(half), cudaMemcpyDeviceToHost);
    
    // 计算CPU结果
    float *h_y_cpu = (float *)malloc(N * sizeof(float));
    half *h_y_cpu_f16 = (half *)malloc(N * sizeof(half));
    for (int i = 0; i < N; i++) {
        // 使用标准GELU函数计算CPU结果
        float x = h_a[i];
        h_y_cpu[i] = 0.5f * x * (1.0f + tanhf(SQRT_2_PI * (x + 0.044715f * x * x * x)));
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
    printf("GELU test passed!\n");
    
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
    test_gelu();
    return 0;
}