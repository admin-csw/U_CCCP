#include <stdio.h>
#include "cuda_runtime.h"

__device__ int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

__global__ void calculateFibonacci(int n) {
    int tid = threadIdx.x;
    int result = fibonacci(n);
    printf("Thread %d: Fibonacci numbrer at position %d is %d\n", tid, n, result);
}

int main() {
    int n = 10; 
    calculateFibonacci<<<1, 1>>>(n);
    cudaDeviceSynchronize();
    return 0;
}