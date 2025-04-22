#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>


#define WARP_SIZE 256
#define WARP_SIZE_S 16
#define PAD 1
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define MAX_EXP_F32 88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f
#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)

// col2row means read x[row][col] and write y[col][row] // 判断连续内存
// row2col means read x[col][row] and write y[row][col]
//  1 2 3               1  4 
//  4 5 6       -- >    2  5       连续内存 x 为 1，2，3，4，5，6 
//                      3  6
__global__ void matrix_transpose_fp32_col2row_kernel(float *x,float *y,int row,int col) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int g_row = tid / col;
    const int g_col = tid % col;
    if (tid < row * col) {
        y[g_col * row + g_row] = x[tid];
    }
}

__global__ void matrix_transpose_fp32_row2col_kernel(float *x,float *y,int row,int col) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int g_col = tid / row;
    const int g_row = tid % row;
    if (tid < row * col) {
        y[tid] = x[g_row * col + g_col];
    }
}

__global__ void matrix_transpose_fp32x4_col2row_kernel(float *x,float *y,int row,int col) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int g_row = (tid * 4) / col;
    const int g_col = (tid*4) % col; 
    if ( g_row < row && (g_col+3) < col) {
        float4 reg_x = FLOAT4(x[tid])
        y[g_col * row + g_row] = reg_x.x;
        y[(g_col+1) * row + g_row] = reg_x.y;
        y[(g_col+1) * row + g_row ] = reg_x.z;
        y[(g_col+1) * row + g_row] = reg_x.w;
    }
}

__global__ void matrix_transpose_fp32x4_row2col_kernel(float *x,float *y,int row,int col) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int g_col = (tid * 4) / row;
    const int g_row = (tid*4) % row;
    if ( g_col < col && (g_row+3) < row) {
        float4 reg_y;
        reg_y.x = x[g_row * col + g_col];
        reg_y.y = x[(g_row+1) * col + g_col];
        reg_y.z = x[(g_row+2) * col + g_col];
        reg_y.w = x[(g_row+3) * col + g_col];
        FLOAT4(y[tid]) = reg_y;
    }
}


__global__ void matrix_transpose_fp32_diag2d_kernel(float *x,float *y,int row,int col){
    const int block_y = blockIdx.x;
    const int block_x = (blockIdx.y + blockIdx.x) % gridDim.x;
    const int global_col = block_x * blockDim.x + threadIdx.x;
    const int global_row = block_y * blockDim.y + threadIdx.y;
//   const int block_y = blockIdx.x;
//   const int block_x = (blockIdx.x + blockIdx.y) % gridDim.x;
//   const int global_col = threadIdx.x + blockDim.x * block_x;
//   const int global_row = threadIdx.y + blockDim.y * block_y;

    if (global_col < col && global_row < row){
        y[global_row * col + global_col] = x[global_col * row + global_row];
    }
}

__global__ void mat_transpose_f32_col2row2d_kernel(float *x, float *y, const int row, const int col) {
  const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (global_x < col && global_y < row) {
    y[global_x * row + global_y] = x[global_y * col + global_x];
  }
}

__global__ void mat_transpose_f32_row2col2d_kernel(float *x, float *y, const int row, const int col) {
    // row2col2d 等价于 col2row2d
  const int global_y = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_x = blockIdx.y * blockDim.y + threadIdx.y;
  if (global_y < col && global_x < row) {
    y[global_y * row + global_x] = x[global_x * col + global_y];
  }
}



void test_matrix_transpose() {
    const int ROW = 32;
    const int COL = 32;
    const int N = ROW * COL;
    
    // 生成测试数据
    float *h_a = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)rand() / RAND_MAX * 10.0f - 5.0f; // 生成-5到5之间的随机数
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 分配设备内存
    float *d_a, *d_y;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    
    // 拷贝数据到设备
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);

    // 调用核函数 - col2row
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    float milliseconds = 0.0f;
    cudaEventRecord(start);
    matrix_transpose_fp32_col2row_kernel<<<grid, block>>>(d_a, d_y, ROW, COL);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Matrix transpose col2row kernel execution time: %f ms\n", milliseconds);

    // 调用核函数 - row2col
    milliseconds = 0.0f;
    cudaEventRecord(start);
    matrix_transpose_fp32_row2col_kernel<<<grid, block>>>(d_a, d_y, ROW, COL);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Matrix transpose row2col kernel execution time: %f ms\n", milliseconds);

    // 调用核函数 - col2row x4
    dim3 block4(64);
    dim3 grid4((N + block.x*4 - 1) / block.x*4);
    milliseconds = 0.0f;
    cudaEventRecord(start);
    matrix_transpose_fp32x4_col2row_kernel<<<grid4, block4>>>(d_a, d_y, ROW, COL);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Matrix transpose col2row x4 kernel execution time: %f ms\n", milliseconds);

    // 调用核函数 - row2col x4
    milliseconds = 0.0f;
    cudaEventRecord(start);
    matrix_transpose_fp32x4_row2col_kernel<<<grid4, block4>>>(d_a, d_y, ROW, COL);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Matrix transpose row2col x4 kernel execution time: %f ms\n", milliseconds);

    // 调用核函数 - diag2d
    dim3 block2d(16, 16);
    dim3 grid2d((COL + block2d.x - 1) / block2d.x, (ROW + block2d.y - 1) / block2d.y);
    milliseconds = 0.0f;
    cudaEventRecord(start);
    matrix_transpose_fp32_diag2d_kernel<<<grid2d, block2d>>>(d_a, d_y, ROW, COL);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Matrix transpose diag2d kernel execution time: %f ms\n", milliseconds);

    // 调用核函数 - col2row2d
    milliseconds = 0.0f;
    cudaEventRecord(start);
    mat_transpose_f32_col2row2d_kernel<<<grid2d, block2d>>>(d_a, d_y, ROW, COL);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Matrix transpose col2row2d kernel execution time: %f ms\n", milliseconds);

    // 调用核函数 - row2col2d
    milliseconds = 0.0f;
    cudaEventRecord(start);
    mat_transpose_f32_row2col2d_kernel<<<grid2d, block2d>>>(d_a, d_y, ROW, COL);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Matrix transpose row2col2d kernel execution time: %f ms\n", milliseconds);

    // 验证结果
    float *h_y = (float *)malloc(N * sizeof(float));
    cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 计算CPU结果
    float *h_y_cpu = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < ROW; i++) {
        for (int j = 0; j < COL; j++) {
            h_y_cpu[j * ROW + i] = h_a[i * COL + j];
        }
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
    printf("Matrix transpose test passed!\n");
    
    // 释放资源
    free(h_a);
    free(h_y);
    free(h_y_cpu);
    cudaFree(d_a);
    cudaFree(d_y);
}

int main() {
    test_matrix_transpose();
    return 0;
}