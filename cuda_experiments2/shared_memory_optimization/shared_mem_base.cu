#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

void __global__ view_matrix(int *a,int *b, int *c,int nrow,int ncol,int mrow,int mcol) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < nrow && j < ncol) {
        c[i * mcol + j] = a[i * ncol + j] + b[i * mcol + j];
    }
}


int main() {
    int nrow, ncol, mrow, mcol;
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size_a, size_b, size_c;
    nrow = 4;
    ncol = 4;
    mrow = 4;
    mcol = 4;
    size_a = nrow * ncol * sizeof(int);
    size_b = mrow * mcol * sizeof(int);
    size_c = nrow * mcol * sizeof(int);
    a = (int *)malloc(size_a);
    b = (int *)malloc(size_b);
    c = (int *)malloc(size_c);
    cudaMalloc((void **)&d_a, size_a);
    cudaMalloc((void **)&d_b, size_b);
    cudaMalloc((void **)&d_c, size_c);
    // initialize matrix a and b with random values
    for (int i = 0; i < nrow; i++) {
        for (int j = 0; j < ncol; j++) {
            a[i * ncol + j] = rand() % 100;
        } 
    }
    for (int i = 0; i < mrow; i++) {
        for (int j = 0; j < mcol; j++) {
            b[i * mcol + j] = rand() % 100;
         } 
    }
    for (int i = 0; i < nrow; i++) {
        for (int j = 0; j < mcol; j++) {
            c[i * mcol + j] = 0;
        } 
    }
    cudaMemcpy(d_a, a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size_b, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, size_c, cudaMemcpyHostToDevice);
    dim3 block(2,2);
    dim3 grid((ncol + block.x - 1) / block.x, (nrow + block.y - 1) / block.y);
    view_matrix<<<grid, block>>>(d_a, d_b, d_c, nrow, ncol, mrow, mcol);
    cudaMemcpy(c, d_c, size_c, cudaMemcpyDeviceToHost);
    for (int i = 0; i < nrow; i++) {
        for (int j = 0; j < mcol; j++) {
            printf("%d ", c[i * mcol + j]);
        }
        printf("\n");
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);
}