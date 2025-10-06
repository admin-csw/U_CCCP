#include <stdio.h>
#include <cuda_runtime.h>

// 기본적인 순차 팩토리얼 (각 스레드가 독립적으로 계산)
__global__ void factorial_parallel(long long* results, int* numbers, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        int num = numbers[idx];
        long long fact = 1;
        
        for (int i = 1; i <= num; i++) {
            fact *= i;
        }
        
        results[idx] = fact;
    }
}

// 리덕션을 이용한 팩토리얼 (하나의 팩토리얼을 여러 스레드가 협력하여 계산)
__global__ void factorial_reduction(long long* result, int n) {
    extern __shared__ long long sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1; // 1부터 시작
    
    // 각 스레드가 자신의 인덱스 값을 로드
    sdata[tid] = (i <= n) ? i : 1;
    __syncthreads();
    
    // 리덕션으로 곱셈 수행
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && (blockIdx.x * blockDim.x + tid + s + 1) <= n) {
            sdata[tid] *= sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 블록의 결과를 글로벌 메모리에 저장
    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

// 최종 결과를 곱하는 커널
__global__ void multiply_blocks(long long* partial_results, long long* final_result, int num_blocks) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        long long result = 1;
        for (int i = 0; i < num_blocks; i++) {
            result *= partial_results[i];
        }
        *final_result = result;
    }
}

// 공유 메모리를 이용한 세그먼트 팩토리얼 (수정된 버전)
__global__ void factorial_segments(long long* results, int* numbers, int n) {
    extern __shared__ long long shared_data[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    if (idx < n) {
        int num = numbers[idx];
        
        // 각 스레드가 독립적으로 팩토리얼 계산 (세그먼트 방식이 아닌 직접 계산)
        long long factorial = 1;
        for (int i = 1; i <= num; i++) {
            factorial *= i;
        }
        
        // 결과를 올바른 위치에 저장
        results[idx] = factorial;
    }
}

// CPU 참조 구현
long long cpu_factorial(int n) {
    long long result = 1;
    for (int i = 1; i <= n; i++) {
        result *= i;
    }
    return result;
}

int main() {
    const int N = 10;  // 계산할 팩토리얼의 개수
    const int BLOCK_SIZE = 256;
    const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    printf("=== GPU 병렬 팩토리얼 계산 ===\n\n");
    
    // 테스트할 숫자들
    int h_numbers[N];
    for (int i = 0; i < N; i++) {
        h_numbers[i] = i + 1;  // 1!, 2!, 3!, ..., 10!
    }
    
    // 호스트 메모리 할당
    long long* h_results1 = (long long*)malloc(N * sizeof(long long));
    long long* h_results3 = (long long*)malloc(N * sizeof(long long));
    
    // 디바이스 메모리 할당
    int* d_numbers;
    long long* d_results1;
    long long* d_results3;
    
    cudaMalloc(&d_numbers, N * sizeof(int));
    cudaMalloc(&d_results1, N * sizeof(long long));
    cudaMalloc(&d_results3, N * sizeof(long long));
    
    cudaMemcpy(d_numbers, h_numbers, N * sizeof(int), cudaMemcpyHostToDevice);
    
    // 성능 측정을 위한 이벤트
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // CPU 참조 계산
    printf("CPU 참조 결과:\n");
    float cpu_time;
    cudaEventRecord(start);
    for (int i = 0; i < N; i++) {
        long long result = cpu_factorial(h_numbers[i]);
        printf("%d! = %lld\n", h_numbers[i], result);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_time, start, stop);
    printf("CPU 실행 시간: %.3f ms\n\n", cpu_time);
    
    // 방법 1: 기본 병렬 팩토리얼 (각 스레드가 독립적으로 계산)
    printf("1. 기본 병렬 팩토리얼:\n");
    cudaEventRecord(start);
    factorial_parallel<<<GRID_SIZE, BLOCK_SIZE>>>(d_results1, d_numbers, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time1;
    cudaEventElapsedTime(&time1, start, stop);
    
    cudaMemcpy(h_results1, d_results1, N * sizeof(long long), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < N; i++) {
        printf("%d! = %lld\n", h_numbers[i], h_results1[i]);
    }
    printf("실행 시간: %.3f ms\n\n", time1);
    
    // 방법 2: 리덕션 기반 팩토리얼 (큰 팩토리얼 하나를 병렬로 계산)
    printf("2. 리덕션 기반 팩토리얼 (15! 계산):\n");
    int large_num = 15;
    int num_blocks = (large_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    long long* d_partial_results;
    long long* d_final_result;
    cudaMalloc(&d_partial_results, num_blocks * sizeof(long long));
    cudaMalloc(&d_final_result, sizeof(long long));
    
    cudaEventRecord(start);
    factorial_reduction<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(long long)>>>(d_partial_results, large_num);
    multiply_blocks<<<1, 1>>>(d_partial_results, d_final_result, num_blocks);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time2;
    cudaEventElapsedTime(&time2, start, stop);
    
    long long final_result;
    cudaMemcpy(&final_result, d_final_result, sizeof(long long), cudaMemcpyDeviceToHost);
    
    printf("%d! = %lld\n", large_num, final_result);
    printf("리덕션 실행 시간: %.3f ms\n", time2);
    printf("CPU 참조: %d! = %lld\n", large_num, cpu_factorial(large_num));
    printf("\n");
    
    // 방법 3: 세그먼트 기반 팩토리얼
    printf("3. 세그먼트 기반 팩토리얼:\n");
    cudaEventRecord(start);
    factorial_segments<<<N, BLOCK_SIZE, BLOCK_SIZE * sizeof(long long)>>>(d_results3, d_numbers, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time3;
    cudaEventElapsedTime(&time3, start, stop);
    
    cudaMemcpy(h_results3, d_results3, N * sizeof(long long), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < N; i++) {
        printf("%d! = %lld\n", h_numbers[i], h_results3[i]);
    }
    printf("실행 시간: %.3f ms\n\n", time3);
    
    // 성능 비교
    printf("=== 성능 비교 ===\n");
    printf("CPU 순차:        %.3f ms\n", cpu_time);
    printf("GPU 기본 병렬:   %.3f ms (%.2fx %s)\n", time1, cpu_time/time1, time1 < cpu_time ? "빠름" : "느림");
    printf("GPU 리덕션:      %.3f ms\n", time2);
    printf("GPU 세그먼트:    %.3f ms (%.2fx %s)\n", time3, cpu_time/time3, time3 < cpu_time ? "빠름" : "느림");
    
    // 메모리 정리
    free(h_results1);
    free(h_results3);
    cudaFree(d_numbers);
    cudaFree(d_results1);
    cudaFree(d_results3);
    cudaFree(d_partial_results);
    cudaFree(d_final_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}