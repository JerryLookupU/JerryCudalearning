#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <tuple>
#include <algorithm>
#include <cuda_runtime.h>



#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])


__global__ void histogram_kernel(int* a,int* y,int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(&y[a[idx]], 1);
    }
}

__global__ void histogram_i32x4_kernel2(int* a,int* y,int n) {
    int idx = 4*(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < n) {
        int4 reg_a = INT4(a[idx]);
        atomicAdd(&y[reg_a.x], 1);
        atomicAdd(&y[reg_a.y], 1);
        atomicAdd(&y[reg_a.z], 1);
        atomicAdd(&y[reg_a.w], 1);
    }
}

void test_histogram() {
    const int N = 1024;
    const int BINS = 10;
    
    // 生成测试数据
    int *h_a = (int *)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        h_a[i] = rand() % BINS;
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 分配设备内存
    int *d_a, *d_y;
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_y, BINS * sizeof(int));
    
    // 拷贝数据到设备
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, BINS * sizeof(int));
    
    // 调用核函数
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    float milliseconds = 0.0f;
    cudaEventRecord(start);
    histogram_kernel<<<grid, block>>>(d_a, d_y, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Histogram kernel execution time: %f ms\n", milliseconds);

    // 验证结果
    int *h_y = (int *)malloc(BINS * sizeof(int));
    memset(h_y, 0, BINS * sizeof(int));
    cudaMemcpy(h_y, d_y, BINS * sizeof(int), cudaMemcpyDeviceToHost);
    
    // 计算CPU结果
    int *h_y_cpu = (int *)malloc(BINS * sizeof(int));
    memset(h_y_cpu, 0, BINS * sizeof(int));
    for (int i = 0; i < N; i++) {
        h_y_cpu[h_a[i]]++;
    }
    
    // 比较结果
    for (int i = 0; i < BINS; i++) {
        if (h_y[i] != h_y_cpu[i]) {
            printf("Error at bin %d: GPU=%d, CPU=%d\n", i, h_y[i], h_y_cpu[i]);
        }
    }
    printf("Histogram test passed!\n");

    // 测试i32x4版本
    cudaMemset(d_y, 0, BINS * sizeof(int));
    milliseconds = 0.0f;
    cudaEventRecord(start);
    histogram_i32x4_kernel2<<<grid, block>>>(d_a, d_y, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Histogram kernel execution time: %f ms\n", milliseconds);

    cudaMemcpy(h_y, d_y, BINS * sizeof(int), cudaMemcpyDeviceToHost);
    
    // 比较结果
    for (int i = 0; i < BINS; i++) {
        if (h_y[i] != h_y_cpu[i]) {
            printf("Error at bin %d (i32x4): GPU=%d, CPU=%d\n", i, h_y[i], h_y_cpu[i]);
        }
    }
    printf("Histogram i32x4 test passed!\n");
    
    // 释放资源
    free(h_a);
    free(h_y);
    free(h_y_cpu);
    cudaFree(d_a);
    cudaFree(d_y);
}

int main() {
    test_histogram();
    return 0;
}
