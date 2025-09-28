#include <iostream>
#include <cuda_runtime.h>

__global__ void addKernel(float* deviceData, float value) {
    deviceData[threadIdx.x] += deviceData[threadIdx.x] + value;
}

int main() {
    int count = 1;
    size_t size = count * sizeof(float);
    float* hostData = new float[count];
    hostData[0] = 2.0f;
    
    cudaHostRegister(hostData, size, cudaHostRegisterDefault);
    
    float* devicePointer;
    cudaHostGetDevicePointer((void**)&devicePointer, hostData, 0);
    
    addKernel<<<1, 1>>>(devicePointer, 5.0f);
    cudaDeviceSynchronize();
    
    std::cout << "Result: " << hostData[0] << std::endl;
    
    cudaHostUnregister(hostData);
    delete[] hostData;
    return 0;
}
