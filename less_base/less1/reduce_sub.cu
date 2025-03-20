#include <iostream>


extern "C" __global__ void reduce_sub(int* input,int* output,int n){
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if(tid < n){
        output[tid] = output[tid] - input[tid];
    }
}


// int main() {
//     const int dim = 4;
//     int *h_input = new int[dim];
//     int *h_output = new int[dim];
//     int *d_input, *d_output;
//     for (int i = 0; i < dim; i++) {
//         h_input[i] = i;
//     }
//     for (int i=0; i<dim; i++){
//         h_output[i] = 1;
//     }
//     cudaMalloc(&d_input, dim*sizeof(int));
//     cudaMalloc(&d_output, dim*sizeof(int));
//     cudaMemcpy(d_input, h_input, dim*sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_output, h_output, dim*sizeof(int), cudaMemcpyHostToDevice);
//     reduce_sub<<<1, dim>>>(d_input, d_output, dim);

//     cudaMemcpy(h_output, d_output, dim*sizeof(int), cudaMemcpyDeviceToHost);

//     for(int i=0; i<dim; i++){
//         std::cout << h_output[i] << std::endl;
//     }   
//     delete[] h_input;
//     delete[] h_output; 
//     cudaFree(d_input);
//     cudaFree(d_output);
  
//     return 0;
// }
