#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel for x + y operation
__global__ void addKernel(int x, int y, int* result) {
    *result = x + y;
}

// CUDA kernel for x * y operation
__global__ void multiplyKernel(int x, int y, int* result) {
    *result = x * y;
}

int main() {
    int x = 5;
    int y = 3;
    int a = 2;
    int b = 4;

    int result_add, result_multiply;

    // GPU device 0
    cudaSetDevice(0);

    int *d_result_add;
    cudaMalloc((void**)&d_result_add, sizeof(int));

    cudaMemcpyToSymbol("x", &x, sizeof(int));
    cudaMemcpyToSymbol("y", &y, sizeof(int));

    addKernel<<<1, 1>>>(x, y, d_result_add);
    cudaMemcpy(&result_add, d_result_add, sizeof(int), cudaMemcpyDeviceToHost);

    // GPU device 1
    cudaSetDevice(1);

    int *d_result_multiply;
    cudaMalloc((void**)&d_result_multiply, sizeof(int));

    cudaMemcpyToSymbol("a", &a, sizeof(int));
    cudaMemcpyToSymbol("b", &b, sizeof(int));

    multiplyKernel<<<1, 1>>>(a, b, d_result_multiply);

    cudaMemcpy(&result_multiply, d_result_multiply, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Result of x + y on GPU 0: %d\n", result_add);
    printf("Result of a * b on GPU 1: %d\n", result_multiply);  

    cudaFree(d_result_add);
    cudaFree(d_result_multiply);
    return 0;
}
