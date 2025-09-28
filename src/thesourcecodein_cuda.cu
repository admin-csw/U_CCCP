#include <stdio.h>
#include <cuda_runtime.h>

#define SIZE 5

__global__ void arrayAdd(const int* a, const int* b, int* c, int size) {
    int i = threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int a[SIZE] = {1, 2, 3, 4, 5};
    int b[SIZE] = {10, 20, 30, 40, 50};
    int c[SIZE];

    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, SIZE * sizeof(int));
    cudaMalloc((void**)&d_b, SIZE * sizeof(int));
    cudaMalloc((void**)&d_c, SIZE * sizeof(int));

    cudaMemcpy(d_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    arrayAdd<<<1, SIZE>>>(d_a, d_b, d_c, SIZE);

    cudaMemcpy(c, d_c, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Resultant array (CUDA):\n");
    printf("{1, 2, 3, 4, 5} + {10, 20, 30, 40, 50} = {%d, %d, %d, %d, %d}\n", c[0], c[1], c[2], c[3], c[4]);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
