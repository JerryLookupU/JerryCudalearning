#include <iostream>
#include <cuda_runtime.h>

template <int kBytes>
struct VecInt {};
template<> struct VecInt<1> { using vec_t = int8_t; };
template<> struct VecInt<2> { using vec_t = int16_t; };
template<> struct VecInt<4> { using vec_t = int; };
template<> struct VecInt<8> { using vec_t = int64_t; };
template<> struct VecInt<16> { using vec_t = int4; };



// `st.global.L1::no_allocate` will be translated into `ST.E.NA.[width]` in SASS
#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define ST_NA_FUNC "st.global.L1::no_allocate"
#else
#define ST_NA_FUNC "st.global"
#endif

template <typename dtype_t>
__device__  __forceinline__ void st_na_global(const dtype_t *ptr, const dtype_t& value) {
    st_na_global(reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t*>(ptr),
                 *reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t*>(&value));
}

template <>
__device__  __forceinline__ void st_na_global(const int *ptr, const int& value) {
    asm volatile(ST_NA_FUNC ".s32 [%0], %1;" ::"l"(ptr), "r"(value));
}

template <>
__device__  __forceinline__ void st_na_global(const int64_t *ptr, const int64_t& value) {
    asm volatile(ST_NA_FUNC ".s64 [%0], %1;" ::"l"(ptr), "l"(value));
}

template <>
__device__  __forceinline__ void st_na_global(const float *ptr, const float& value) {
    asm volatile(ST_NA_FUNC ".f32 [%0], %1;" ::"l"(ptr), "f"(value));
}

template <>
__device__  __forceinline__ void st_na_global(const int4 *ptr, const int4& value) {
    asm volatile(ST_NA_FUNC ".v4.s32 [%0], {%1, %2, %3, %4};"
            ::"l"(ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
}


// struct int4 {
//     int x, y, z, w;
// };
// make_int4(1, 2, 3, 4) -> {1, 2, 3, 4} 是cuda 提供内置辅助函数 用于创建int4类型的结构体
// 这个函数可以用于在CUDA设备上创建一个int4类型的结构体实例，而不需要手动分配内存并设置每个成员的值。

__global__ void test_st_na_global(int* int_ptr, int64_t* int64_ptr, float* float_ptr, int4* int4_ptr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        // 测试int类型
        int int_val = 123;
        st_na_global(int_ptr, int_val);
        
        // 测试int64_t类型
        int64_t int64_val = 4567890123;
        st_na_global(int64_ptr, int64_val);
        
        // 测试float类型
        float float_val = 3.14159f;
        st_na_global(float_ptr, float_val);
        
        // 测试int4类型
        int4 int4_val = make_int4(1, 2, 3, 4);
        st_na_global(int4_ptr, int4_val);
    }
}

int main() {
    // 分配设备内存
    int* d_int;
    int64_t* d_int64;
    float* d_float;
    int4* d_int4;
    cudaMalloc(&d_int, sizeof(int));
    cudaMalloc(&d_int64, sizeof(int64_t));
    cudaMalloc(&d_float, sizeof(float));
    cudaMalloc(&d_int4, sizeof(int4));
    
    // 启动测试内核
    test_st_na_global<<<1, 1>>>(d_int, d_int64, d_float, d_int4);
    
    // 同步设备
    cudaDeviceSynchronize();
    
    // 验证结果(这里只是示例，实际测试应该将数据拷贝回主机验证)
    printf("Test functions launched. Use CUDA profiler to verify PTX instructions.\n");
    
    // 释放内存
    cudaFree(d_int);
    cudaFree(d_int64);
    cudaFree(d_float);
    cudaFree(d_int4);
    return 0;
}

// & "D:\workpace\miniforge\Library\nsight-compute\2025.1.1\host\target-windows-x64\nsys.exe" profile --trace=cuda "D:\code\jerrycudalearning\less_base\less9\st_na_ptx.exe"
// & "D:\workpace\miniforge\Library\nsight-compute\2025.1.1\host\target-windows-x64\nsys.exe" stats .\report1.nsys-rep --force-export

// - ST_NA_FUNC 宏定义了是否使用 no_allocate 指令
// - 使用 asm volatile 直接嵌入PTX指令
// - nsys 命令用于性能分析和验证PTX指令