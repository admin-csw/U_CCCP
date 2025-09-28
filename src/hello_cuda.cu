#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello_kernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from thread %d!\n", idx);
}

int main() {
    printf("U-CCCP CUDA Project\n");
    printf("===================\n");
    
    // GPU 정보 출력
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Number of CUDA devices: %d\n", deviceCount);
    
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("Device 0: %s\n", prop.name);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    }
    
    // 간단한 커널 실행
    printf("\nExecuting CUDA kernel...\n");
    hello_kernel<<<2, 4>>>();
    cudaDeviceSynchronize();
    
    return 0;
}
