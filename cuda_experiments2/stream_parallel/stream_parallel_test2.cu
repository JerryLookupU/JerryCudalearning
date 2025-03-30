#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// 多GPU流并行测试函数
void testMultiGPUStreamParallel() {
    // 获取GPU数量
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    
    // 为每个GPU创建流和分配内存
    cudaStream_t* streams = new cudaStream_t[numDevices];
    float** d_A = new float*[numDevices];
    float** d_B = new float*[numDevices];
    float** d_C = new float*[numDevices];
    
    // 设置数组大小
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    
    // 分配主机内存
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    
    // 初始化主机数组
    for (int i = 0; i < numDevices; ++i) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaMalloc((void **)&d_A[i], size);
        cudaMalloc((void **)&d_B[i], size);
        cudaMalloc((void **)&d_C[i], size);
    }
    
    // 数据分块处理
    int elementsPerDevice = numElements / numDevices;
    size_t sizePerDevice = size / numDevices;
    
    // 在每个GPU上并行执行
    for (int i = 0; i < numDevices; ++i) {
        cudaSetDevice(i);
        int offset = i * elementsPerDevice;
        
        // 异步传输和计算
        cudaMemcpyAsync(d_A[i], h_A + offset, sizePerDevice, cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_B[i], h_B + offset, sizePerDevice, cudaMemcpyHostToDevice, streams[i]);
        
        // 内核调用
        vectorAdd<<<ceil(elementsPerDevice/256.0), 256, 0, streams[i]>>>(
            d_A[i], d_B[i], d_C[i], elementsPerDevice);
            
        // 异步回传结果
        cudaMemcpyAsync(h_C + offset, d_C[i], sizePerDevice, cudaMemcpyDeviceToHost, streams[i]);
    }
    
    // 同步所有流
    for (int i = 0; i < numDevices; ++i) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
    }
    
    // 验证结果
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            printf("Error at index %d\n", i);
            break;
        }
    }
    printf("Test passed!\n");
    
    // 释放资源
    for (int i = 0; i < numDevices; ++i) {
        cudaSetDevice(i);
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_C[i]);
        cudaStreamDestroy(streams[i]);
    }
    free(h_A);
    free(h_B);
    free(h_C);
    delete[] streams;
    delete[] d_A;
    delete[] d_B;
    delete[] d_C;
}

int main() {
    testMultiGPUStreamParallel();
    return 0;
}