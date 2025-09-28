#include <stdio.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "TheEmployeesSalary.h"

cudaError_t thehelperfunc(const double* tarray, double* tnewSalaries, int* tSIZE, int* threadsPerBlock, int* blocksPerGrid);

__global__ void TaskDoer(const double* array, double* newSalaries, int* SIZE) {
    int ID = blockIdx.x * blockDim.x + threadIdx.x;
    if (ID < *SIZE) {
        newSalaries[ID] = array[ID] + (array[ID] * 15/100) + 5000;
    }
}

int main() {
    int size = sizeof(TheArrayOfSalaries) / sizeof(TheArrayOfSalaries[0]);

    double newSalaries[size];

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    cudaError_t cudaStatus = thehelperfunc(TheArrayOfSalaries, newSalaries, &size, &threadsPerBlock, &blocksPerGrid);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "The Helper function failed!");
        return 1;
    }


    for (int i = 0; i < size; i++) {
        printf("Employee %d: New Salary = %.2f\n", i + 1, newSalaries[i]);
    }

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Cuda Device Reset failed!");
        return 1;
    }

    return 0;
}

cudaError_t thehelperfunc(const double* tarray, double* tnewSalaries, int* tSIZE, int* threadsPerBlock, int* blocksPerGrid) {
    double* deviceArray = 0;
    double* deviceNewSalaries = 0;
    int* deviceSize = 0;

    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Cuda Set Device failed!  Do you have a CUDA-capable GPU installed?");
        goto AnError;
    }

    cudaStatus = cudaMalloc((void**)&deviceArray, sizeof(double) * (*tSIZE));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Cuda Malloc failed!");
        goto AnError;
    }

    cudaStatus = cudaMalloc((void**)&deviceNewSalaries, sizeof(double) * *tSIZE);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Cuda Malloc failed!");
        goto AnError;
    }

    cudaStatus = cudaMalloc((void**)&deviceSize, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Cuda Malloc failed!");
        goto AnError;
    }

    cudaStatus = cudaMemcpy(deviceArray, TheArrayOfSalaries, sizeof(double) * *tSIZE, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Cuda Memcpy failed!");
        goto AnError;
    }

    cudaStatus = cudaMemcpy(deviceSize, tSIZE, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Cuda Memcpy failed!");
        goto AnError;
    }

    cudaStatus = cudaMemcpy(deviceNewSalaries, tnewSalaries, sizeof(double) * *tSIZE, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Cuda Memcpy failed!");
        goto AnError;
    }

    TaskDoer<<<*blocksPerGrid, *threadsPerBlock>>>(deviceArray, deviceNewSalaries, deviceSize);

    cudaStatus = cudaGetLastError();

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "TaskDoer Kernel failed!");
        goto AnError;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Cuda Device Synchronize failed!");
        goto AnError;
    }

    cudaStatus = cudaMemcpy(tnewSalaries, deviceNewSalaries, sizeof(double) * *tSIZE, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Cuda Memcpy failed!");
        goto AnError;
    }

AnError:
    cudaFree(deviceArray);
    cudaFree(deviceNewSalaries);
    cudaFree(deviceSize);

    return cudaStatus;
}