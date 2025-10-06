#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

__global__ void predicatedKernel(int* input, int* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size; i += stride)
    {
        // Complex predicated execution
        int value = input[i];
        
        // Multiple complex conditions
        bool condition1 = (value % 7 == 0);
        bool condition2 = (value % 11 == 0);
        bool condition3 = (value % 13 == 0);
        bool condition4 = (value > 50 && value < 80);
        bool condition5 = ((value * value) % 17 == 3);
        
        // Predicated execution - all threads compute all results
        int result = value;
        result = condition1 ? (result * 3) : result;
        result = condition2 ? (result + 25) : result;
        result = condition3 ? (result / 2 + 10) : result;
        result = condition4 ? (result * 2) : result;
        result = condition5 ? (result - 15) : result;
        
        output[i] = result;
    }
}

__global__ void nonPredicatedKernel(int* input, int* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size; i += stride)
    {
        // Complex non-predicated execution with multiple branches
        int value = input[i];
        int result = value;
        
        // Multiple if statements causing warp divergence
        if (value % 7 == 0) {
            result = result * 3;
        }
        
        if (value % 11 == 0) {
            result = result + 25;
        }
        
        if (value % 13 == 0) {
            result = result / 2 + 10;
        }
        
        if (value > 50 && value < 80) {
            result = result * 2;
        }
        
        if ((value * value) % 17 == 3) {
            result = result - 15;
        }
        
        output[i] = result;
    }
}

int main() {
    const int SIZE = 1024 * 1024 * 10; // 10M elements
    const int BLOCK_SIZE = 256;
    const int GRID_SIZE = (SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    printf("=== 복잡한 조건문을 사용한 Predicated vs Non-Predicated Kernel 비교 ===\n\n");
    
    // 호스트 메모리 할당
    int* h_input = (int*)malloc(SIZE * sizeof(int));
    int* h_output1 = (int*)malloc(SIZE * sizeof(int));
    int* h_output2 = (int*)malloc(SIZE * sizeof(int));
    
    // 디바이스 메모리 할당
    int *d_input, *d_output1, *d_output2;
    cudaMalloc(&d_input, SIZE * sizeof(int));
    cudaMalloc(&d_output1, SIZE * sizeof(int));
    cudaMalloc(&d_output2, SIZE * sizeof(int));
    
    // 입력 데이터 초기화
    for (int i = 0; i < SIZE; i++) {
        h_input[i] = rand() % 100;
    }
    
    cudaMemcpy(d_input, h_input, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    
    // 출력 메모리 초기화
    cudaMemset(d_output1, 0, SIZE * sizeof(int));
    cudaMemset(d_output2, 0, SIZE * sizeof(int));
    
    // 성능 측정용 이벤트
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 1. Predicated Kernel 테스트
    printf("1. Predicated Kernel 실행...\n");
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        predicatedKernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_input, d_output1, SIZE);
        cudaDeviceSynchronize(); // 커널 완료 대기
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // 커널 실행 에러 확인
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Predicated Kernel 에러: %s\n", cudaGetErrorString(err));
    }
    
    float time1;
    cudaEventElapsedTime(&time1, start, stop);
    
    // 2. Non-Predicated Kernel 테스트
    printf("2. Non-Predicated Kernel 실행...\n");
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        nonPredicatedKernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_input, d_output2, SIZE);
        cudaDeviceSynchronize(); // 커널 완료 대기
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // 커널 실행 에러 확인
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Non-Predicated Kernel 에러: %s\n", cudaGetErrorString(err));
    }
    
    float time2;
    cudaEventElapsedTime(&time2, start, stop);
    
    // 결과 확인
    cudaMemcpy(h_output1, d_output1, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output2, d_output2, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    
    // 결과 검증
    bool match = true;
    for (int i = 0; i < 10; i++) {
        if (h_output1[i] != h_output2[i]) {
            match = false;
            break;
        }
    }
    
    printf("\n=== 결과 ===\n");
    printf("Predicated Kernel:     %.3f ms\n", time1);
    printf("Non-Predicated Kernel: %.3f ms\n", time2);
    printf("성능 차이:             %.2fx\n", time2 / time1);
    printf("결과 일치:             %s\n", match ? "Yes" : "No");
    
    // 샘플 결과 출력
    printf("\n샘플 입력/출력 (처음 10개):\n");
    for (int i = 0; i < 10; i++) {
        printf("입력: %2d -> 출력: %2d\n", h_input[i], h_output1[i]);
    }
    
    // 메모리 정리
    free(h_input);
    free(h_output1);
    free(h_output2);
    cudaFree(d_input);
    cudaFree(d_output1);
    cudaFree(d_output2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}