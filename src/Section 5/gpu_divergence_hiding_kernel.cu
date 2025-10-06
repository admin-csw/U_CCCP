
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

// Divergence hiding kernel - 분기 없이 조건부 실행
__global__ void divergenceHidingKernel(int* input, int* output, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < size; i += stride) {
        int value = input[i];
        
        // Branch-free computation - 모든 스레드가 같은 경로 실행
        int evenResult = value * 2;
        int oddResult = value;

        bool isEven = (value % 2 == 0);
        output[i] = isEven ? evenResult : oddResult;
    }
}

// Branching kernel - 명시적 분기문 사용
__global__ void branchingKernel(int* input, int* output, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < size; i += stride) {
        int value = input[i];
        
        // Explicit branching - warp divergence 유발
        if (value % 2 == 0) {
            output[i] = value * 2;  // 짝수 처리
        } else {
            output[i] = value;      // 홀수 처리
        }
    }
}

// 복잡한 분기를 사용하는 커널
__global__ void complexBranchingKernel(int* input, int* output, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < size; i += stride) {
        int value = input[i];
        int result;
        
        // 복잡한 분기 로직으로 심각한 warp divergence 유발
        if (value % 4 == 0) {
            result = value * 3;
            // 추가 계산
            for (int j = 0; j < 3; j++) {
                result += j;
            }
        } else if (value % 4 == 1) {
            result = value + 10;
            result = result * result % 100;
        } else if (value % 4 == 2) {
            result = value / 2;
            if (result > 20) {
                result -= 5;
            }
        } else { // value % 4 == 3
            result = value;
            for (int j = 0; j < 2; j++) {
                result = (result + j) % 50;
            }
        }
        
        output[i] = result;
    }
}

int main() {
    const int SIZE = 1024 * 1024 * 8; // 8M elements
    const int BLOCK_SIZE = 256;
    const int GRID_SIZE = (SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    printf("=== Divergence Hiding vs Branching Kernel 성능 비교 ===\n\n");
    
    // 호스트 메모리 할당
    int* h_input = (int*)malloc(SIZE * sizeof(int));
    int* h_output1 = (int*)malloc(SIZE * sizeof(int));
    int* h_output2 = (int*)malloc(SIZE * sizeof(int));
    int* h_output3 = (int*)malloc(SIZE * sizeof(int));
    
    // 디바이스 메모리 할당
    int *d_input, *d_output1, *d_output2, *d_output3;
    cudaMalloc(&d_input, SIZE * sizeof(int));
    cudaMalloc(&d_output1, SIZE * sizeof(int));
    cudaMalloc(&d_output2, SIZE * sizeof(int));
    cudaMalloc(&d_output3, SIZE * sizeof(int));
    
    // 입력 데이터 초기화
    srand(42);
    for (int i = 0; i < SIZE; i++) {
        h_input[i] = rand() % 100;
    }
    
    cudaMemcpy(d_input, h_input, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    
    // 성능 측정용 이벤트
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 1. Divergence Hiding Kernel 테스트
    printf("1. Divergence Hiding Kernel 실행...\n");
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        divergenceHidingKernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_input, d_output1, SIZE);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time1;
    cudaEventElapsedTime(&time1, start, stop);
    
    // 2. Simple Branching Kernel 테스트
    printf("2. Simple Branching Kernel 실행...\n");
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        branchingKernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_input, d_output2, SIZE);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time2;
    cudaEventElapsedTime(&time2, start, stop);
    
    // 3. Complex Branching Kernel 테스트
    printf("3. Complex Branching Kernel 실행...\n");
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        complexBranchingKernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_input, d_output3, SIZE);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time3;
    cudaEventElapsedTime(&time3, start, stop);
    
    // 결과 확인
    cudaMemcpy(h_output1, d_output1, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output2, d_output2, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output3, d_output3, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    
    // 결과 검증 (첫 번째와 두 번째 커널 비교)
    bool match = true;
    for (int i = 0; i < 100; i++) {
        if (h_output1[i] != h_output2[i]) {
            match = false;
            break;
        }
    }
    
    printf("\n=== 성능 결과 ===\n");
    printf("Divergence Hiding:     %.3f ms\n", time1);
    printf("Simple Branching:      %.3f ms\n", time2);
    printf("Complex Branching:     %.3f ms\n", time3);
    printf("\n성능 비교:\n");
    printf("Simple Branching vs Hiding:  %.2fx\n", time2 / time1);
    printf("Complex Branching vs Hiding: %.2fx\n", time3 / time1);
    printf("Complex vs Simple Branching: %.2fx\n", time3 / time2);
    printf("\nHiding과 Simple Branching 결과 일치: %s\n", match ? "Yes" : "No");
    
    // 샘플 결과 출력
    printf("\n샘플 결과 (처음 8개):\n");
    printf("입력 | Hiding | Simple | Complex\n");
    printf("----|--------|--------|--------\n");
    for (int i = 0; i < 8; i++) {
        printf("%4d | %6d | %6d | %7d\n", 
               h_input[i], h_output1[i], h_output2[i], h_output3[i]);
    }
    
    // 메모리 정리
    free(h_input);
    free(h_output1);
    free(h_output2);
    free(h_output3);
    cudaFree(d_input);
    cudaFree(d_output1);
    cudaFree(d_output2);
    cudaFree(d_output3);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}