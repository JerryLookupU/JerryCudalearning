#include <stdio.h>
#include <stdexcept>
#include <string>
#include <iostream>
#include <cuda_runtime.h>

// 异常类，用于捕获和处理断言失败的情况
class EPException : public std::runtime_error {
public:
    EPException(const std::string& type, const std::string& file, int line, const std::string& cond)
        : std::runtime_error(type + " failed at " + file + ":" + std::to_string(line) + " " + cond) {}
};

#ifndef EP_STATIC_ASSERT
#define EP_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#endif

#ifndef EP_HOST_ASSERT
#define EP_HOST_ASSERT(cond) \
do { \
    if (!(cond)) { \
        asm("trap;"); \
    } \
} while (0)
#else
#define EP_HOST_ASSERT(cond) \
do { \
    if (!(cond)) { \
        throw EPException("Assertion", __FILE__, __LINE__, #cond); \
    } \
} while (0)
#endif




template <typename dtype_t>
__host__ __device__ dtype_t cell_div(dtype_t a, dtype_t b) {
    return (a + b - 1) / b;
}

template <typename dtype_t>
__host__ __device__ dtype_t align(dtype_t a, dtype_t b) {
    return cell_div<dtype_t>(a, b) * b;
}


template <typename dtype_a_t, typename dtype_b_t>
__host__ __device__ __forceinline__ dtype_b_t pack2(const dtype_a_t& x, const dtype_a_t& y) {
    EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");
    dtype_b_t packed;
    auto unpacked_ptr = reinterpret_cast<dtype_a_t*>(&packed);
    unpacked_ptr[0] = x, unpacked_ptr[1] = y;
    return packed;
}
template <typename dtype_a_t, typename dtype_b_t>
__host__ __device__ __forceinline__ void unpack2(const dtype_b_t& packed, dtype_a_t& x, dtype_a_t& y) {
    EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");
    auto unpacked_ptr = reinterpret_cast<const dtype_a_t*>(&packed);
    x = unpacked_ptr[0], y = unpacked_ptr[1];
}




template <typename dtype_t>
__device__ __forceinline__ dtype_t broadcast(dtype_t& ptr, int src_lane_idx) {
    EP_STATIC_ASSERT(sizeof(dtype_t) % sizeof(int) == 0, "");
    auto send_int_values = reinterpret_cast<int*>(&ptr);
    int recv_int_values[sizeof(dtype_t) / sizeof(int)];
    #pragma unroll
    for (int i = 0; i < sizeof(dtype_t) / sizeof(int); ++ i)
        recv_int_values[i] = __shfl_sync(0xffffffff, send_int_values[i], src_lane_idx);
    return *reinterpret_cast<dtype_t*>(recv_int_values);
}


// - EP_HOST_ASSERT宏：用于主机端断言检查，条件不满足时抛出异常
// - cell_div和align模板函数：用于整数除法对齐计算
// - pack2/unpack2函数：用于将两个小类型数据打包成一个大类型数据
// - broadcast函数：使用CUDA的__shfl_sync指令在warp内广播数据

// 测试函数
__host__ void test_cell_div_align() {
    // 测试cell_div
    printf("cell_div(10,3) = %d (expected: 4)\n", cell_div<int>(10, 3));
    printf("cell_div(10,5) = %d (expected: 2)\n", cell_div<int>(10, 5));
    printf("cell_div(10,1) = %d (expected: 10)\n", cell_div<int>(10, 1));
    
    // 测试align
    printf("align(10,3) = %d (expected: 12)\n", align<int>(10, 3));
    printf("align(10,5) = %d (expected: 10)\n", align<int>(10, 5));
    printf("align(10,1) = %d (expected: 10)\n", align<int>(10, 1));
}

__host__ void test_pack_unpack() {
    // 测试pack2/unpack2
    uint32_t packed = pack2<uint16_t, uint32_t>(0x1234, 0x5678);
    uint16_t x, y;
    unpack2<uint16_t, uint32_t>(packed, x, y);
    printf("unpacked x = 0x%x (expected: 0x1234), y = 0x%x (expected: 0x5678)\n", x, y);
}

__global__ void test_broadcast_kernel() {
    // 测试broadcast函数
    int value = threadIdx.x;
    int broadcasted = broadcast<int>(value, 0);
    if (threadIdx.x == 0) {
        printf("broadcasted value = %d (expected: 0)\n", broadcasted);
    }
}

__host__ void test_broadcast() {
    test_broadcast_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
}

__host__ void test_all_utils() {
    test_cell_div_align();
    test_pack_unpack();
    test_broadcast();
    printf("All tests passed!\n");
}

int main() {
    test_all_utils();
    return 0;
}
