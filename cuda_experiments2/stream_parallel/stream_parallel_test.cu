// steam_parallel_text.cu
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// 简单的并行计算内核函数
__global__ void vectorAdd(const float* A, const float* B, float* C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

// 流并行测试函数
void testStreamParallel() {
    // 设置数组大小
    int numElements = 50000;
    size_t size = numElements * sizeof(float);

    // 分配主机内存
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // 初始化主机数组
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // 分配设备内存
    float *d_A1, *d_B1, *d_C1;
    float *d_A2, *d_B2, *d_C2;
    cudaMalloc((void **)&d_A1, size);
    cudaMalloc((void **)&d_B1, size);
    cudaMalloc((void **)&d_C1, size);
    cudaMalloc((void **)&d_A2, size);
    cudaMalloc((void **)&d_B2, size);
    cudaMalloc((void **)&d_C2, size);

    // 创建CUDA流
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // 分块处理数据
    int halfElements = numElements / 2;
    size_t halfSize = size / 2;

    // 在流1上执行第一部分
    cudaMemcpyAsync(d_A1, h_A, halfSize, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_B1, h_B, halfSize, cudaMemcpyHostToDevice, stream1);
    vectorAdd<<<ceil(halfElements/256.0), 256, 0, stream1>>>(d_A1, d_B1, d_C1, halfElements);
    cudaMemcpyAsync(h_C, d_C1, halfSize, cudaMemcpyDeviceToHost, stream1);

    // 在流2上执行第二部分
    cudaMemcpyAsync(d_A2, h_A + halfElements, halfSize, cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_B2, h_B + halfElements, halfSize, cudaMemcpyHostToDevice, stream2);
    vectorAdd<<<ceil(halfElements/256.0), 256, 0, stream2>>>(d_A2, d_B2, d_C2, halfElements);
    cudaMemcpyAsync(h_C + halfElements, d_C2, halfSize, cudaMemcpyDeviceToHost, stream2);

    // 同步流
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // 验证结果
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            printf("Error at index %d\n", i);
            break;
        }
    }
    printf("Test passed!\n");

    // 释放资源
    cudaFree(d_A1);
    cudaFree(d_B1);
    cudaFree(d_C1);
    cudaFree(d_A2);
    cudaFree(d_B2);
    cudaFree(d_C2);
    free(h_A);
    free(h_B);
    free(h_C);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}

int main() {
    testStreamParallel();
    return 0;
}