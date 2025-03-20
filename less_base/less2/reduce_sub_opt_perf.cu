#include <iostream>
#include <locale>
#include <cuda_runtime.h>
#include <vector>

// 声明不同展开次数的内核
extern "C" __global__ void reduce_sub(int* input, int* output, int n);
extern "C" __global__ void reduce_sub8(int* input, int* output, int n);
extern "C" __global__ void reduce_sub16(int* input, int* output, int n);
extern "C" __global__ void reduce_sub32(int* input, int* output, int n);


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

  // 调整for循环内代码块的缩进，确保逻辑清晰
  for (auto size : test_sizes) {
      int *d_input, *d_output;
      size_t bytes = size * sizeof(int);
      
      CHECK(cudaMalloc(&d_input, bytes));
      CHECK(cudaMalloc(&d_output, bytes));
      
      CHECK(cudaMemset(d_input, 1, bytes));
      CHECK(cudaMemset(d_output, 0, bytes));
  
      // 测试不同版本的内核
      const std::vector<std::pair<void(*)(int*,int*,int), std::string>> kernels = {
          {reduce_sub, "4 rows"},
          {reduce_sub8, "8 rows"},
          {reduce_sub16, "16 rows"},
          {reduce_sub32, "32 rows"}
      };
  
      for (const auto& kernel_pair : kernels) {
          auto kernel_func = kernel_pair.first;
          auto kernel_name = kernel_pair.second;
          for (auto block_size : block_sizes) {
        float total_time = 0;
        int grid_size = (size + block_size - 1) / block_size;
  
        kernel_func<<<grid_size, block_size>>>(d_input, d_output, size);
        CHECK(cudaDeviceSynchronize());
  
        for (int t = 0; t < trials; ++t) {
          CHECK(cudaEventRecord(start));
          kernel_func<<<grid_size, block_size>>>(d_input, d_output, size);
          CHECK(cudaEventRecord(stop));
          CHECK(cudaEventSynchronize(stop));
          
          float milliseconds = 0;
          CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
          total_time += milliseconds;
        }
  
        float avg_time = total_time / trials;
        float bandwidth = (2.0f * bytes * 1e-6) / (avg_time / 1e3);
        
        std::cout << kernel_name << " | Block Size: " << block_size << " | Data Volume: " << size 
                  << " | Average Time: " << avg_time << "ms"
                  << " | Bandwidth: " << bandwidth << "GB/s\n";
          }
      }
      
      
  
      CHECK(cudaFree(d_input));
      CHECK(cudaFree(d_output));
      d_input = nullptr;
      d_output = nullptr;
  }

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    cudaDeviceReset();
  }

  int main() {
    std::locale::global(std::locale(""));
    std::wcout.imbue(std::locale());
    test_performance();
    return 0;
  }
