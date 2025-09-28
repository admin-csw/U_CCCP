#include <stdio.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <iostream>

__global__ void stage1Kernel(int* input, int* intermediate, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        intermediate[tid] += input[tid] + 1;
    }
}

__global__ void stage2Kernel(int* intermediate, int* output, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        output[tid] += intermediate[tid] * 2;
    }
}

int main() {
    const int size = 1000;

    // Allocate host memory
    int *h_input = (int*)malloc(size * sizeof(int));
    int *h_output = (int*)malloc(size * sizeof(int));

    // Initialize host data
    for (int i = 0; i < size; i++) {
        h_input[i] = i;
        h_output[i] = 0;
    }

    // Allocate device memory
    int *d_input, *d_intermediate, *d_output;
    cudaMalloc((void**)&d_input, size * sizeof(int));
    cudaMalloc((void**)&d_intermediate, size * sizeof(int));
    cudaMalloc((void**)&d_output, size * sizeof(int));

    //Create CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Transfer data from host to device using stream1
    cudaMemcpyAsync(d_input, h_input, size * sizeof(int), cudaMemcpyHostToDevice, stream1);

    // Launch stage1Kernel in stream1
    dim3 blockDim(256);
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x);

    // Launch stage1Kernel in stream1
    stage1Kernel<<<gridDim, blockDim, 0, stream1>>>(d_input, d_intermediate, size);

    // stream2가 stream1의 완료를 기다리도록 이벤트 사용
    cudaEvent_t stage1_complete;
    cudaEventCreate(&stage1_complete);
    cudaEventRecord(stage1_complete, stream1);
    cudaStreamWaitEvent(stream2, stage1_complete, 0);

    // Launch stage2Kernel in stream2 (stage1 완료 후)
    stage2Kernel<<<gridDim, blockDim, 0, stream2>>>(d_intermediate, d_output, size);

    // Transfer data from device to host using stream2
    cudaMemcpyAsync(h_output, d_output, size * sizeof(int), cudaMemcpyDeviceToHost, stream2);

    // Synchronize streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Cleanup event
    cudaEventDestroy(stage1_complete);

    // Print results
    for (int i = 0; i < size; i++) {
       std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_intermediate);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    
    return 0;
}