#include <stdio.h>
#include <cuda_runtime.h>

#define NUM_GPUS 2

// kernel ufnction to perform computatio on GPU 0
__global__ void kernelA(int *x) {
    *x = (*x) * (*x);
}

// kernel function to perform computation on GPU 1
__global__ void kernelB(int *y) {
    *y = (*y) * 3;
}


int main() {
    int h_data_0; // Host data for GPU 0
    int h_data_1; // Host data for GPU 1
    int *d_data_0; // Device data for GPU 0
    int *d_data_1; // Device data for GPU 1

    cudaStream_t stream0, stream1;

    cudaSetDevice(0);
    cudaStreamCreate(&stream0);
    cudaMalloc(&d_data_0, sizeof(int));

    cudaSetDevice(1);
    cudaStreamCreate(&stream1);
    cudaMalloc(&d_data_1, sizeof(int));

    // Enable peer access between GPUs
    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0);

    //  Initialize data on GPU 0
    h_data_0 = 2;
    cudaMemcpy(d_data_0, &h_data_0, sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel on GPU 0
    kernelA<<<1, 1, 0, stream0>>>(d_data_0);

    // Copy result from GPU 0 to GPU 1
    cudaSetDevice(1);
    cudaMemcpyPeerAsync(d_data_1, 1, d_data_0, 0, sizeof(int));

    // Launch kernel on GPU 1
    kernelB<<<1, 1, 0, stream1>>>(d_data_1);

    // Copy result back to host from GPU 1 to CPU
    cudaMemcpy(&h_data_1, d_data_1, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Result: %d\n", h_data_1); // Should print 12 (2^2 * 3 = 12)

    // Free device memory
    cudaSetDevice(0);
    cudaFree(d_data_0);
    cudaSetDevice(1);
    cudaFree(d_data_1);

    // Destroy streams
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);

    return 0;
}