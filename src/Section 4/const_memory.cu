#include <stdio.h>
#include <cuda_runtime.h>


__constant__ int constData;

__global__ void myKernel() {
    int data = constData;   
    printf("Const Data: %d\n", constData);
}

int main() {
     int hostData = 100;
    cudaMemcpyToSymbol(constData, &hostData, sizeof(int));

    myKernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    return 0;
}