#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// 初始化多GPU设备并启用P2P访问
void initMultiGPU(int ngpus) {
    for (int i = 0; i < ngpus; i++) {
        CHECK(cudaSetDevice(i));
        
        // 检查设备是否支持P2P访问
        int canAccessPeer;
        for (int j = 0; j < ngpus; j++) {
            if (i == j) continue;
            CHECK(cudaDeviceCanAccessPeer(&canAccessPeer, i, j));
            if (canAccessPeer) {
                CHECK(cudaDeviceEnablePeerAccess(j, 0));
                printf("GPU%d can access GPU%d\n", i, j);
            }
        }
    }
}

// 使用统一内存进行数据传输
void unifiedMemoryTransfer(int ngpus, size_t size) {
    // 分配统一内存
    float *d_data;
    CHECK(cudaMallocManaged(&d_data, size * sizeof(float)));
    
    // 初始化数据
    for (int i = 0; i < size; i++) {
        d_data[i] = i;
    }
    
    // 在多个GPU上处理数据
    for (int i = 0; i < ngpus; i++) {
        CHECK(cudaSetDevice(i));
        
        // 预取数据到当前GPU
        CHECK(cudaMemPrefetchAsync(d_data, size * sizeof(float), i));
        
        // 简单的内核处理
        // 这里可以添加实际的处理内核
        
        printf("GPU%d processed data\n", i);
    }
    
    CHECK(cudaFree(d_data));
}

// 使用P2P进行直接数据传输
void peerToPeerTransfer(int ngpus, size_t size) {
    float *d_src, *d_dst;
    
    // 在GPU0上分配源数据
    CHECK(cudaSetDevice(0));
    CHECK(cudaMalloc(&d_src, size * sizeof(float)));
    
    // 在GPU1上分配目标数据
    CHECK(cudaSetDevice(1));
    CHECK(cudaMalloc(&d_dst, size * sizeof(float)));
    
    // 初始化源数据
    CHECK(cudaSetDevice(0));
    float *h_data = (float *)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        h_data[i] = i;
    }
    CHECK(cudaMemcpy(d_src, h_data, size * sizeof(float), cudaMemcpyHostToDevice));
    free(h_data);
    
    // 直接从GPU0拷贝到GPU1
    CHECK(cudaMemcpyPeer(d_dst, 1, d_src, 0, size * sizeof(float)));
    printf("Data transferred from GPU0 to GPU1 via P2P\n");
    
    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_dst));
}

// GPU间数据传输和求和
void gpuDataTransferAndSum(int ngpus, size_t size) {
    int *a_input, *b_input, *c_output;
    
    // 在GPU0上分配和初始化a_input
    CHECK(cudaSetDevice(0));
    CHECK(cudaMalloc(&a_input, size * sizeof(int)));
    
    // 初始化数据
    int *h_data = (int *)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        h_data[i] = i;
    }
    CHECK(cudaMemcpy(a_input, h_data, size * sizeof(int), cudaMemcpyHostToDevice));
    free(h_data);
    
    // 在GPU1上分配b_input和c_output
    CHECK(cudaSetDevice(1));
    CHECK(cudaMalloc(&b_input, size * sizeof(int)));
    CHECK(cudaMalloc(&c_output, sizeof(int)));
    
    // 使用P2P将数据从GPU0传输到GPU1
    CHECK(cudaMemcpyPeer(b_input, 1, a_input, 0, size * sizeof(int)));
    
    // 在GPU1上启动求和内核
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    
    // 简单的求和内核
    __global__ void sumKernel(int *input, int *output, int size) {
        __shared__ int sdata[256];
        
        int tid = threadIdx.x;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        
        // 加载数据到共享内存
        if (i < size) {
            sdata[tid] = input[i];
        } else {
            sdata[tid] = 0;
        }
        __syncthreads();
        
        // 归约求和
        for (int s = blockDim.x/2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        
        // 第一个线程写入结果
        if (tid == 0) {
            atomicAdd(output, sdata[0]);
        }
    }
    
    // 重置输出值
    CHECK(cudaMemset(c_output, 0, sizeof(int)));
    
    // 启动内核
    sumKernel<<<grid, block>>>(b_input, c_output, size);
    CHECK(cudaDeviceSynchronize());
    
    // 读取结果
    int result;
    CHECK(cudaMemcpy(&result, c_output, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Sum result on GPU1: %d\n", result);
    
    // 释放资源
    CHECK(cudaFree(a_input));
    CHECK(cudaFree(b_input));
    CHECK(cudaFree(c_output));
}

int main() {
    int ngpus;
    CHECK(cudaGetDeviceCount(&ngpus));
    printf("Found %d GPUs\n", ngpus);
    
    if (ngpus < 2) {
        printf("Need at least 2 GPUs for this demo\n");
        return 1;
    }
    
    // 初始化多GPU环境
    initMultiGPU(ngpus);
    
    // 测试统一内存传输
    printf("\nTesting Unified Memory:\n");
    unifiedMemoryTransfer(ngpus, 1024);
    
    // 测试P2P直接传输
    printf("\nTesting Peer-to-Peer Transfer:\n");
    peerToPeerTransfer(ngpus, 1024);
    
    // 测试GPU间数据传输和求和
    printf("\nTesting GPU Data Transfer and Sum:\n");
    gpuDataTransferAndSum(ngpus, 1024);
    
    // 禁用P2P访问
    for (int i = 0; i < ngpus; i++) {
        CHECK(cudaSetDevice(i));
        for (int j = 0; j < ngpus; j++) {
            if (i != j) {
                cudaDeviceDisablePeerAccess(j);
            }
        }
    }
    
    return 0;
}