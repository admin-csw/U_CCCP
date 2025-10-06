#include <stdio.h>
#include "cuda_runtime.h"

__device__ int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}


__global__ void calculateFactorial(int* array, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size) {
        int num = array[tid];
        int result = factorial(num);
        printf("Thread %d: Factorial of %d is %d\n", tid, num, result);
    }
}

int main() {
    int size = 5;
    int h_array[] = {5, 6, 7, 8, 9};
    int* d_array;

    cudaMalloc((void**)&d_array, size * sizeof(int));
    cudaMemcpy(d_array, h_array, size * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blockPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
 
    calculateFactorial<<<blockPerGrid, threadsPerBlock>>>(d_array, size);
    cudaDeviceSynchronize();
    cudaFree(d_array);
    return 0;
}