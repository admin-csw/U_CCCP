#include <stdio.h>

#define N 1024
__global__ void kernel(cudaTextureObject_t tex) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    float x = tex1Dfetch<float>(tex, i);
    printf("Thread %d: %f\n", i, x);
}

void call_kernel(cudaTextureObject_t tex) {
    kernel<<<1, 256>>>(tex);
}

int main() {
    // declare and allocate memory
    float *buffer;
    cudaMalloc((void**)&buffer, N * sizeof(float));
    
    // Initialize buffer with test data
    float *host_buffer = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        host_buffer[i] = (float)i * 4.0f;
    }
    cudaMemcpy(buffer, host_buffer, N * sizeof(float), cudaMemcpyHostToDevice);

    // create texture object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = buffer;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32; // 32-bit float
    resDesc.res.linear.sizeInBytes = N * sizeof(float);

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    // create texture object: we only have to this once!
    cudaTextureObject_t tex = 0;
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

    // Set the texture object with the buffer data
    call_kernel(tex);
    
    // Wait for kernel completion
    cudaDeviceSynchronize();

    // destroy texture object and free memory
    cudaDestroyTextureObject(tex);
    cudaFree(buffer);
    free(host_buffer);

    return 0;

}
