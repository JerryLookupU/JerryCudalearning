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
    
    // 分配设备内存
    CHECK(cudaMalloc(&d_input, bytes));
    CHECK(cudaMalloc(&d_output, bytes));
    
    // 初始化数据
    CHECK(cudaMemset(d_input, 1, bytes));
    CHECK(cudaMemset(d_output, 0, bytes));

    for (auto block_size : block_sizes) {
      float total_time = 0;
      int grid_size = (size + block_size - 1) / block_size;

      // 预热
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

      // 计算指标
      float avg_time = total_time / trials;
      float bandwidth = (2.0f * bytes * 1e-6) / (avg_time / 1e3);
      
      std::cout << "Size: " << size 
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
// Size: 1024 | Block: 256 | Time: 0.0046848ms | Bandwidth: 1748.63GB/s
// Size: 1024 | Block: 512 | Time: 0.0114656ms | Bandwidth: 714.485GB/s
// Size: 4096 | Block: 256 | Time: 0.0096896ms | Bandwidth: 3381.77GB/s
// Size: 4096 | Block: 512 | Time: 0.0075424ms | Bandwidth: 4344.51GB/s
// Size: 16384 | Block: 256 | Time: 0.0103872ms | Bandwidth: 12618.6GB/s
// Size: 16384 | Block: 512 | Time: 0.0098016ms | Bandwidth: 13372.5GB/s
// Size: 65536 | Block: 256 | Time: 0.0089856ms | Bandwidth: 58347.6GB/s
// Size: 65536 | Block: 512 | Time: 0.009872ms | Bandwidth: 53108.6GB/s
// Size: 262144 | Block: 256 | Time: 0.0115072ms | Bandwidth: 182247GB/s
// Size: 262144 | Block: 512 | Time: 0.011216ms | Bandwidth: 186979GB/s
// Size: 1048576 | Block: 256 | Time: 0.0093824ms | Bandwidth: 894079GB/s
// Size: 1048576 | Block: 512 | Time: 0.0100096ms | Bandwidth: 838056GB/s