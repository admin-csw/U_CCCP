#include <stdio.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

__global__ void myKernel(int* data1) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    data1[tid] += 1;
}

__global__ void myKernel2(int* data2) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    data2[tid] *= 2;
}

int main() {
    int dataSize = 1024;
    int blockSize = 256;
    int gridSize = dataSize / blockSize;

    int* hostData1;
    int* hostData2;
    int* deviceData1;
    int* deviceData2;

    // Allocate host and device memory for data1
    hostData1 = (int*)malloc(dataSize * sizeof(int));
    cudaMalloc((void**)&deviceData1, dataSize * sizeof(int));

    // Allocate host and device memory for data2
    hostData2 = (int*)malloc(dataSize * sizeof(int));
    cudaMalloc((void**)&deviceData2, dataSize * sizeof(int));

    // Initialize host data1
    for (int i = 0; i < dataSize; i++) {
        hostData1[i] = i;
    }

    // Initialize host data2
    for (int i = 0; i < dataSize; i++) {
        hostData2[i] = i;
    }

    // Create CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Asynchronously copy data1 to device
    cudaMemcpyAsync(deviceData1, hostData1, dataSize * sizeof(int), cudaMemcpyHostToDevice, stream1);

    // Launch kernel1 in stream1
    myKernel<<<gridSize, blockSize, 0, stream1>>>(deviceData1);

    // Asynchronously copy data2 to device
    cudaMemcpyAsync(deviceData2, hostData2, dataSize * sizeof(int), cudaMemcpyHostToDevice, stream2);

    // Launch kernel2 in stream2
    myKernel2<<<gridSize, blockSize, 0, stream2>>>(deviceData2);

    // Asynchronously copy results back to host
    cudaMemcpyAsync(hostData1, deviceData1, dataSize * sizeof(int), cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(hostData2, deviceData2, dataSize * sizeof(int), cudaMemcpyDeviceToHost, stream2);

    // Synchronize streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Print the modified data1
    printf("Data1 after kernel execution:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", hostData1[i]);
    }
    printf("\n");

    // Print the modified data2
    printf("Data2 after kernel execution:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", hostData2[i]);
    }
    printf("\n");

    // Clean up
    cudaFree(deviceData1);
    cudaFree(deviceData2);
    free(hostData1);
    free(hostData2);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return 0;
}