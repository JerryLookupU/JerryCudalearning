#include <iostream>
#include <cuda_runtime.h>

extern "C" __global__ void reduce_sub(int* input, int* output, int n) {
    __shared__ int s_data[256];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_idx = threadIdx.x;
    
    // 使用共享内存缓存输入数据
    if (tid < n) {
        s_data[local_idx] = input[tid];
    }
    __syncthreads();
    
    // 展开四路计算减少分支
    if (tid < n) {
        int val = output[tid];
        #pragma unroll 4
        for(int offset = 0; offset < blockDim.x; offset += warpSize) {
            val -= s_data[(local_idx + offset) % blockDim.x];
        }
        output[tid] = val;
    }
}

extern "C" __global__ void reduce_sub8(int* input, int* output, int n) {
    __shared__ int s_data[256];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_idx = threadIdx.x;
    
    // 使用共享内存缓存输入数据
    if (tid < n) {
        s_data[local_idx] = input[tid];
    }
    __syncthreads();
    
    // 展开8路计算减少分支
    if (tid < n) {
        int val = output[tid];
        #pragma unroll 8
        for(int offset = 0; offset < blockDim.x; offset += warpSize) {
            val -= s_data[(local_idx + offset) % blockDim.x];
        }
        output[tid] = val;
    }
}

extern "C" __global__ void reduce_sub16(int* input, int* output, int n) {
    __shared__ int s_data[256];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_idx = threadIdx.x;
    
    // 使用共享内存缓存输入数据
    if (tid < n) {
        s_data[local_idx] = input[tid];
    }
    __syncthreads();
    
    // 展开16路计算减少分支
    if (tid < n) {
        int val = output[tid];
        #pragma unroll 16
        for(int offset = 0; offset < blockDim.x; offset += warpSize) {
            val -= s_data[(local_idx + offset) % blockDim.x];
        }
        output[tid] = val;
    }
}

extern "C" __global__ void reduce_sub32(int* input, int* output, int n) {
    __shared__ int s_data[256];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_idx = threadIdx.x;
    
    // 使用共享内存缓存输入数据
    if (tid < n) {
        s_data[local_idx] = input[tid];
    }
    __syncthreads();
    
    // 展开32路计算减少分支
    if (tid < n) {
        int val = output[tid];
        #pragma unroll 32
        for(int offset = 0; offset < blockDim.x; offset += warpSize) {
            val -= s_data[(local_idx + offset) % blockDim.x];
        }
        output[tid] = val;
    }
}

// int main() {
//     const int dim = 4;
//     int *h_input, *h_output;
//     int *d_input, *d_output;

//     // 分配主机内存
//     h_input = new int[dim];
//     h_output = new int[dim];

//     // 初始化输入数据
//     for (int i = 0; i < dim; i++) {
//         h_input[i] = i;
//         h_output[i] = 1;
//     }

//     // 分配设备内存
//     cudaMalloc(&d_input, dim * sizeof(int));
//     cudaMalloc(&d_output, dim * sizeof(int));

//     // 数据传输
//     cudaMemcpy(d_input, h_input, dim * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_output, h_output, dim * sizeof(int), cudaMemcpyHostToDevice);

//     // 启动内核
//     int blockSize = 32;
//     int gridSize = (dim + blockSize - 1) / blockSize;
//     cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
//     reduce_sub<<<gridSize, blockSize>>>(d_input, d_output, dim);

//     // 取回结果
//     cudaMemcpy(h_output, d_output, dim * sizeof(int), cudaMemcpyDeviceToHost);

//     // 输出结果
//     for (int i = 0; i < dim; i++) {
//         std::cout << "Result[" << i << "] = " << h_output[i] << std::endl;
//     }

//     // 清理资源
//     delete[] h_input;
//     delete[] h_output;
//     cudaFree(d_input);
//     cudaFree(d_output);

//     return 0;
// }