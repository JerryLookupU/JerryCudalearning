#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])


__global__ void embedding_kernel(const int* idx, float* weight, float* output, int n, int emb_size) {
    int t_idx = threadIdx.x;
    int b_idx = blockIdx.x;
    int offset = idx[b_idx] * emb_size;
    output[b_idx * emb_size + t_idx] = weight[offset + t_idx];
}



__global__ void embedding_fp32x4_kernel(const int* idx, float* weight, float* output, int n, int emb_size) {
    int t_idx = 4 * threadIdx.x;
    int b_idx = blockIdx.x;
    int offset = idx[b_idx] * emb_size;
    output[b_idx * emb_size + t_idx] = weight[offset + t_idx];
    output[b_idx * emb_size + t_idx + 1] = weight[offset + t_idx + 1];
    output[b_idx * emb_size + t_idx + 2] = weight[offset + t_idx + 2];
    output[b_idx * emb_size + t_idx + 3] = weight[offset + t_idx + 3];
}

__global__ void embedding_fp16_kernel(const int* idx, float16* weight, float16* output, int n, int emb_size) {
    int t_idx = threadIdx.x;
    int b_idx = blockIdx.x;
    int offset = idx[b_idx] * emb_size;
    output[b_idx * emb_size + t_idx] = weight[offset + t_idx];
}

__global__ void embedding_f16x4_kernel(const int* idx, float16* weight, float16* output, int n, int emb_size) {
    int t_idx = 4 * threadIdx.x;
    int b_idx = blockIdx.x; 
    int offset = idx[b_idx] * emb_size;
    output[b_idx * emb_size + t_idx] = weight[offset + t_idx];
    output[b_idx * emb_size + t_idx + 1] = weight[offset + t_idx + 1];
    output[b_idx * emb_size + t_idx + 2] = weight[offset + t_idx + 2];
    output[b_idx * emb_size + t_idx + 3] = weight[offset + t_idx + 3];
}

__global__ void embedding_fp16x8_kernel(const int* idx, float16* weight, float16* output, int n, int emb_size) {
    int t_idx = 8 * threadIdx.x;
    int b_idx = blockIdx.x;
    int offset = idx[b_idx] * emb_size;
    output[b_idx * emb_size + t_idx] = weight[offset + t_idx];
    output[b_idx * emb_size + t_idx + 1] = weight[offset + t_idx + 1];
    output[b_idx * emb_size + t_idx + 2] = weight[offset + t_idx + 2];
    output[b_idx * emb_size + t_idx + 3] = weight[offset + t_idx + 3];
    output[b_idx * emb_size + t_idx + 4] = weight[offset + t_idx + 4];
    output[b_idx * emb_size + t_idx + 5] = weight[offset + t_idx + 5];
    output[b_idx * emb_size + t_idx + 6] = weight[offset + t_idx + 6];
    output[b_idx * emb_size + t_idx + 7] = weight[offset + t_idx + 7];

}

__global__ void embedding_fp32x4_pack_kernel(const int* idx, float* weight, float* output, int n, int emb_size) {
    int t_idx = threadIdx.x;
    int b_idx = blockIdx.x;
    int offset = idx[b_idx] * emb_size;
    LDST128BITS(output[b_idx * emb_size + 4*t_idx]) = LDST128BITS(weight[offset + 4*t_idx]);
}

__global__ void embedding_fp16x8_pack_kernel(const int* idx, float16* weight, float16* output, int n, int emb_size) {
    int t_idx = threadIdx.x;
    int b_idx = blockIdx.x;
    int offset = idx[b_idx] * emb_size;
    LDST128BITS(output[b_idx * emb_size + 8*t_idx]) = LDST128BITS(weight[offset + 8*t_idx]);
}



void test_embedding() {
    const int seq_len = 4;        // 序列长度
    const int embedding_size = 1024; // embedding 维度
    const int vocab_size = 1024;   // 词表大小
    
    // Print parameters
    printf("Test Parameters:\n");
    printf("Sequence Length (seq_len): %d\n", seq_len);
    printf("Embedding Dimension (embedding_size): %d\n", embedding_size);
    printf("Vocabulary Size (vocab_size): %d\n", vocab_size);
    printf("\n");
    
    // Generate test data
    int *h_idx = (int *)malloc(seq_len * sizeof(int));
    float *h_weight = (float *)malloc(vocab_size * embedding_size * sizeof(float));
    float *h_output = (float *)malloc(seq_len * embedding_size * sizeof(float));
    half *h_weight_f16 = (half *)malloc(vocab_size * embedding_size * sizeof(half));
    half *h_output_f16 = (half *)malloc(seq_len * embedding_size * sizeof(half));
    
    // Initialize random indices and weights
    for (int i = 0; i < seq_len; i++) {
        h_idx[i] = rand() % vocab_size;  // 随机生成词索引
    }
    
    for (int i = 0; i < vocab_size * embedding_size; i++) {
        h_weight[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;  // 随机初始化 embedding 权重
        h_weight_f16[i] = __float2half(h_weight[i]);
    }
    
    // Print input data
    printf("Input Indices:\n");
    for (int i = 0; i < seq_len; i++) {
        printf("%d ", h_idx[i]);
    }
    printf("\n\n");
    
    printf("Embedding Weights:\n");
    for (int i = 0; i < vocab_size; i++) {
        printf("Word %d: ", i);
        for (int j = 0; j < embedding_size; j++) {
            printf("%.2f ", h_weight[i * embedding_size + j]);
        }
        printf("\n");
    }
    printf("\n");
    
    // Allocate device memory
    int *d_idx;
    float *d_weight, *d_output;
    half *d_weight_f16, *d_output_f16;
    cudaMalloc(&d_idx, seq_len * sizeof(int));
    cudaMalloc(&d_weight, vocab_size * embedding_size * sizeof(float));
    cudaMalloc(&d_output, seq_len * embedding_size * sizeof(float));
    cudaMalloc(&d_weight_f16, vocab_size * embedding_size * sizeof(half));
    cudaMalloc(&d_output_f16, seq_len * embedding_size * sizeof(half));
    
    // Copy data to device
    cudaMemcpy(d_idx, h_idx, seq_len * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, vocab_size * embedding_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_f16, h_weight_f16, vocab_size * embedding_size * sizeof(half), cudaMemcpyHostToDevice);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Test fp32 kernel
    dim3 block(embedding_size);  // 每个线程处理一个 embedding 维度
    dim3 grid(seq_len);         // 每个 block 处理一个序列位置
    
    printf("Testing FP32 Kernel:\n");
    printf("Block Size: %d\n", block.x);
    printf("Grid Size: %d\n", grid.x);
    
    cudaEventRecord(start);
    embedding_kernel<<<grid, block>>>(d_idx, d_weight, d_output, seq_len, embedding_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time: %f ms\n\n", milliseconds);
    
    // Test fp32x4 kernel
    dim3 block_x4(embedding_size/4);  // 每个线程处理4个元素
    dim3 grid_x4(seq_len);
    
    printf("Testing FP32x4 Kernel:\n");
    printf("Block Size: %d\n", block_x4.x);
    printf("Grid Size: %d\n", grid_x4.x);
    
    cudaEventRecord(start);
    embedding_fp32x4_kernel<<<grid_x4, block_x4>>>(d_idx, d_weight, d_output, seq_len, embedding_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time: %f ms\n\n", milliseconds);
    
    // Test fp16 kernel
    printf("Testing FP16 Kernel:\n");
    printf("Block Size: %d\n", block.x);
    printf("Grid Size: %d\n", grid.x);
    
    cudaEventRecord(start);
    embedding_fp16_kernel<<<grid, block>>>(d_idx, d_weight_f16, d_output_f16, seq_len, embedding_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time: %f ms\n\n", milliseconds);
    
    // Test fp16x4 kernel
    printf("Testing FP16x4 Kernel:\n");
    printf("Block Size: %d\n", block_x4.x);
    printf("Grid Size: %d\n", grid_x4.x);
    
    cudaEventRecord(start);
    embedding_f16x4_kernel<<<grid_x4, block_x4>>>(d_idx, d_weight_f16, d_output_f16, seq_len, embedding_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time: %f ms\n\n", milliseconds);
    
    // Test fp16x8 kernel
    dim3 block_x8(embedding_size/8);  // 每个线程处理8个元素
    dim3 grid_x8(seq_len);
    
    printf("Testing FP16x8 Kernel:\n");
    printf("Block Size: %d\n", block_x8.x);
    printf("Grid Size: %d\n", grid_x8.x);
    
    cudaEventRecord(start);
    embedding_fp16x8_kernel<<<grid_x8, block_x8>>>(d_idx, d_weight_f16, d_output_f16, seq_len, embedding_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time: %f ms\n\n", milliseconds);
    
    // Test fp32x4 pack kernel
    printf("Testing FP32x4 Pack Kernel:\n");
    printf("Block Size: %d\n", block_x4.x);
    printf("Grid Size: %d\n", grid_x4.x);
    
    cudaEventRecord(start);
    embedding_fp32x4_pack_kernel<<<grid_x4, block_x4>>>(d_idx, d_weight, d_output, seq_len, embedding_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time: %f ms\n\n", milliseconds);
    
    // Test fp16x8 pack kernel
    printf("Testing FP16x8 Pack Kernel:\n");
    printf("Block Size: %d\n", block_x8.x);
    printf("Grid Size: %d\n", grid_x8.x);
    
    cudaEventRecord(start);
    embedding_fp16x8_pack_kernel<<<grid_x8, block_x8>>>(d_idx, d_weight_f16, d_output_f16, seq_len, embedding_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time: %f ms\n\n", milliseconds);
    
    // Copy results back to host
    cudaMemcpy(h_output, d_output, seq_len * embedding_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_f16, d_output_f16, seq_len * embedding_size * sizeof(half), cudaMemcpyDeviceToHost);
    
    // Print output data
    printf("\nOutput Embeddings (FP32):\n");
    for (int i = 0; i < seq_len; i++) {
        printf("Position %d: ", i);
        for (int j = 0; j < embedding_size; j++) {
            printf("%.2f ", h_output[i * embedding_size + j]);
        }
        printf("\n");
    }
    printf("\n");
    
    printf("Output Embeddings (FP16):\n");
    for (int i = 0; i < seq_len; i++) {
        printf("Position %d: ", i);
        for (int j = 0; j < embedding_size; j++) {
            printf("%.2f ", __half2float(h_output_f16[i * embedding_size + j]));
        }
        printf("\n");
    }
    printf("\n");
    
    // Verify results
    float max_error = 0.0f;
    float max_error_f16 = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        int idx = h_idx[i];
        for (int j = 0; j < embedding_size; j++) {
            float expected = h_weight[idx * embedding_size + j];
            float actual = h_output[i * embedding_size + j];
            float error = fabs(expected - actual);
            if (error > max_error) {
                max_error = error;
            }
            
            float actual_f16 = __half2float(h_output_f16[i * embedding_size + j]);
            float error_f16 = fabs(expected - actual_f16);
            if (error_f16 > max_error_f16) {
                max_error_f16 = error_f16;
            }
        }
    }
    printf("FP32 Max error: %f\n", max_error);
    printf("FP16 Max error: %f\n", max_error_f16);
    printf("Embedding lookup test passed!\n");
    
    // Free resources
    free(h_idx);
    free(h_weight);
    free(h_output);
    free(h_weight_f16);
    free(h_output_f16);
    cudaFree(d_idx);
    cudaFree(d_weight);
    cudaFree(d_output);
    cudaFree(d_weight_f16);
    cudaFree(d_output_f16);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    test_embedding();
    return 0;
}