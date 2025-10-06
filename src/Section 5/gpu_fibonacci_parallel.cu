#include <stdio.h>
#include <cuda_runtime.h>

// 반복적인 피보나치 계산 (각 스레드가 하나의 피보나치 수 계산)
__global__ void fibonacci_parallel(int* results, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        if (idx == 0) {
            results[idx] = 0;
        } else if (idx == 1) {
            results[idx] = 1;
        } else {
            // 각 스레드가 자신의 피보나치 수를 반복문으로 계산
            int a = 0, b = 1, c;
            for (int i = 2; i <= idx; i++) {
                c = a + b;
                a = b;
                b = c;
            }
            results[idx] = b;
        }
    }
}

// 매트릭스 곱셈을 이용한 피보나치 (더 효율적인 병렬 방법)
__device__ void matrix_multiply(long long F[2][2], long long M[2][2]) {
    long long x = F[0][0] * M[0][0] + F[0][1] * M[1][0];
    long long y = F[0][0] * M[0][1] + F[0][1] * M[1][1];
    long long z = F[1][0] * M[0][0] + F[1][1] * M[1][0];
    long long w = F[1][0] * M[0][1] + F[1][1] * M[1][1];
    
    F[0][0] = x;
    F[0][1] = y;
    F[1][0] = z;
    F[1][1] = w;
}

__device__ void matrix_power(long long F[2][2], int n) {
    if (n == 0 || n == 1) return;
    
    long long M[2][2] = {{1, 1}, {1, 0}};
    matrix_power(F, n / 2);
    matrix_multiply(F, F);
    
    if (n % 2 != 0) {
        matrix_multiply(F, M);
    }
}

__global__ void fibonacci_matrix_parallel(long long* results, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        if (idx == 0) {
            results[idx] = 0;
        } else if (idx == 1) {
            results[idx] = 1;
        } else {
            long long F[2][2] = {{1, 1}, {1, 0}};
            matrix_power(F, idx - 1);
            results[idx] = F[0][0];
        }
    }
}

// 공유 메모리를 사용한 피보나치 (블록 내 협력)
__global__ void fibonacci_shared_parallel(int* results, int n) {
    extern __shared__ int shared_fib[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // 공유 메모리 초기화
    if (tid == 0) shared_fib[0] = 0;
    if (tid == 1 && block_size > 1) shared_fib[1] = 1;
    
    __syncthreads();
    
    // 블록 내에서 순차적으로 피보나치 계산
    for (int i = 2; i < block_size && (blockIdx.x * blockDim.x + i) < n; i++) {
        if (tid == i) {
            shared_fib[i] = shared_fib[i-1] + shared_fib[i-2];
        }
        __syncthreads();
    }
    
    // 결과를 글로벌 메모리에 저장
    if (idx < n && tid < block_size) {
        results[idx] = shared_fib[tid];
    }
}

int main() {
    const int N = 20;  // 계산할 피보나치 수의 개수
    const int BLOCK_SIZE = 256;
    const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    printf("=== GPU 병렬 피보나치 수열 계산 ===\n\n");
    
    // 호스트 메모리 할당
    int* h_results1 = (int*)malloc(N * sizeof(int));
    long long* h_results2 = (long long*)malloc(N * sizeof(long long));
    int* h_results3 = (int*)malloc(N * sizeof(int));
    
    // 디바이스 메모리 할당
    int* d_results1;
    long long* d_results2;
    int* d_results3;
    
    cudaMalloc(&d_results1, N * sizeof(int));
    cudaMalloc(&d_results2, N * sizeof(long long));
    cudaMalloc(&d_results3, N * sizeof(int));
    
    // 성능 측정을 위한 이벤트
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 방법 1: 기본 병렬 피보나치
    printf("1. 기본 병렬 피보나치 (각 스레드가 독립적으로 계산):\n");
    cudaEventRecord(start);
    fibonacci_parallel<<<GRID_SIZE, BLOCK_SIZE>>>(d_results1, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time1;
    cudaEventElapsedTime(&time1, start, stop);
    
    cudaMemcpy(h_results1, d_results1, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < N; i++) {
        printf("F(%d) = %d\n", i, h_results1[i]);
    }
    printf("실행 시간: %.3f ms\n\n", time1);
    
    // 방법 2: 매트릭스 기반 병렬 피보나치 (큰 수에 효율적)
    printf("2. 매트릭스 기반 병렬 피보나치 (O(log n) 알고리즘):\n");
    cudaEventRecord(start);
    fibonacci_matrix_parallel<<<GRID_SIZE, BLOCK_SIZE>>>(d_results2, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time2;
    cudaEventElapsedTime(&time2, start, stop);
    
    cudaMemcpy(h_results2, d_results2, N * sizeof(long long), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < N; i++) {
        printf("F(%d) = %lld\n", i, h_results2[i]);
    }
    printf("실행 시간: %.3f ms\n\n", time2);
    
    // 방법 3: 공유 메모리 기반 협력 계산
    printf("3. 공유 메모리 기반 협력 계산:\n");
    cudaEventRecord(start);
    fibonacci_shared_parallel<<<1, min(N, BLOCK_SIZE), BLOCK_SIZE * sizeof(int)>>>(d_results3, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time3;
    cudaEventElapsedTime(&time3, start, stop);
    
    cudaMemcpy(h_results3, d_results3, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < min(N, BLOCK_SIZE); i++) {
        printf("F(%d) = %d\n", i, h_results3[i]);
    }
    printf("실행 시간: %.3f ms\n\n", time3);
    
    // 성능 비교
    printf("=== 성능 비교 ===\n");
    printf("기본 병렬:     %.3f ms\n", time1);
    printf("매트릭스 기반: %.3f ms\n", time2);
    printf("공유 메모리:   %.3f ms\n", time3);
    
    // 메모리 정리
    free(h_results1);
    free(h_results2);
    free(h_results3);
    cudaFree(d_results1);
    cudaFree(d_results2);
    cudaFree(d_results3);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}