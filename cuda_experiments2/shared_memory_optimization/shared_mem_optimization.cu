#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

// 基本矩阵乘法（未优化共享内存）
__global__ void matrixMulBasic(float *C, float *A, float *B, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        float sum = 0.0f;
        for (int k = 0; k < width; ++k) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

// 使用共享内存的矩阵乘法（可能有bank冲突）
__global__ void matrixMulShared(float *C, float *A, float *B, int width) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int m = 0; m < width / TILE_SIZE; ++m) {
        As[ty][tx] = A[row * width + (m * TILE_SIZE + tx)];
        Bs[ty][tx] = B[(m * TILE_SIZE + ty) * width + col];
        
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < width && col < width) {
        C[row * width + col] = sum;
    }
}

// 优化后的共享内存矩阵乘法（减少bank冲突）
__global__ void matrixMulSharedOptimized(float *C, float *A, float *B, int width) {
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];  // 添加padding避免bank冲突
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int m = 0; m < width / TILE_SIZE; ++m) {
        As[ty][tx] = A[row * width + (m * TILE_SIZE + tx)];
        Bs[ty][tx] = B[(m * TILE_SIZE + ty) * width + col];
        
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < width && col < width) {
        C[row * width + col] = sum;
    }
}

void initMatrix(float *matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i * size + j] = (float)rand() / RAND_MAX;
        }
    }
}

int main() {
    int width = 1024;
    size_t size = width * width * sizeof(float);
    
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    
    initMatrix(h_A, width);
    initMatrix(h_B, width);
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((width + TILE_SIZE - 1) / TILE_SIZE, (width + TILE_SIZE - 1) / TILE_SIZE);
    
    // 测试基本矩阵乘法
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    matrixMulBasic<<<dimGrid, dimBlock>>>(d_C, d_A, d_B, width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Basic matrix multiplication: %.3f ms\n", milliseconds);
    
    // 测试共享内存矩阵乘法（可能有bank冲突）
    cudaEventRecord(start);
    matrixMulShared<<<dimGrid, dimBlock>>>(d_C, d_A, d_B, width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Shared memory matrix multiplication (potential bank conflicts): %.3f ms\n", milliseconds);
    
    // 测试优化后的共享内存矩阵乘法
    cudaEventRecord(start);
    matrixMulSharedOptimized<<<dimGrid, dimBlock>>>(d_C, d_A, d_B, width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Optimized shared memory matrix multiplication (reduced bank conflicts): %.3f ms\n", milliseconds);
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    free(h_A);
    free(h_B);
    free(h_C);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}