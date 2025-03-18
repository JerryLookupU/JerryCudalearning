#include <iostream>
#include <cuda_runtime.h>
#include <vector>

extern "C" __global__ void reduce_sub(int* input, int* output, int n);

#define CHECK(call) {\
  const cudaError_t err = call;\
  if (err != cudaSuccess) {\
    printf("%s in %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__);\
    exit(EXIT_FAILURE);\
  }\
}

void test_performance() {
  const std::vector<int> test_sizes = {1024, 4096, 16384, 65536, 262144, 1048576};
  const std::vector<int> block_sizes = {256, 512};
  const int trials = 10;

  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));

  for (auto size : test_sizes) {
    int *d_input, *d_output;
    size_t bytes = size * sizeof(int);
    
    CHECK(cudaMalloc(&d_input, bytes));
    CHECK(cudaMalloc(&d_output, bytes));
    
    CHECK(cudaMemset(d_input, 1, bytes));
    CHECK(cudaMemset(d_output, 0, bytes));

    for (auto block_size : block_sizes) {
      float total_time = 0;
      int grid_size = (size + block_size - 1) / block_size;

      reduce_sub<<<grid_size, block_size>>>(d_input, d_output, size);
      CHECK(cudaDeviceSynchronize());

      for (int t = 0; t < trials; ++t) {
        CHECK(cudaEventRecord(start));
        reduce_sub<<<grid_size, block_size>>>(d_input, d_output, size);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        total_time += milliseconds;
      }

      float avg_time = total_time / trials;
      float bandwidth = (2.0f * bytes * 1e-6) / (avg_time / 1e3);
      
      std::cout << "Optimized Version | Size: " << size 
                << " | Block: " << block_size
                << " | Time: " << avg_time << "ms"
                << " | Bandwidth: " << bandwidth << "GB/s\n";
    }

    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output));
  }

  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));
}

int main() {
  test_performance();
  return 0;
}

// Optimized Version | Size: 1024 | Block: 256 | Time: 0.0101024ms | Bandwidth: 810.896GB/s
// Optimized Version | Size: 1024 | Block: 512 | Time: 0.0114816ms | Bandwidth: 713.489GB/s
// Optimized Version | Size: 4096 | Block: 256 | Time: 0.0058816ms | Bandwidth: 5571.27GB/s
// Optimized Version | Size: 4096 | Block: 512 | Time: 0.009536ms | Bandwidth: 3436.24GB/s
// Optimized Version | Size: 16384 | Block: 256 | Time: 0.0045024ms | Bandwidth: 29111.6GB/s
// Optimized Version | Size: 16384 | Block: 512 | Time: 0.0096288ms | Bandwidth: 13612.5GB/s
// Optimized Version | Size: 65536 | Block: 256 | Time: 0.0067072ms | Bandwidth: 78167.9GB/s
// Optimized Version | Size: 65536 | Block: 512 | Time: 0.0076192ms | Bandwidth: 68811.4GB/s
// Optimized Version | Size: 262144 | Block: 256 | Time: 0.0100096ms | Bandwidth: 209514GB/s
// Optimized Version | Size: 262144 | Block: 512 | Time: 0.0117056ms | Bandwidth: 179158GB/s
// Optimized Version | Size: 1048576 | Block: 256 | Time: 0.0195424ms | Bandwidth: 429252GB/s
// Optimized Version | Size: 1048576 | Block: 512 | Time: 0.0328352ms | Bandwidth: 255476GB/s