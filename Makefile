# U-CCCP CUDA Project Makefile
NVCC = /usr/local/cuda-11.8/bin/nvcc
CUDA_FLAGS = -arch=sm_61 -std=c++11 --allow-unsupported-compiler
INCLUDES = -I/usr/local/cuda-11.8/include
LIBS = -L/usr/local/cuda-11.8/lib64 -lcudart

SRCDIR = src
BINDIR = bin
BUILDDIR = build

# 소스 파일들
SOURCES = $(wildcard $(SRCDIR)/*.cu)
OBJECTS = $(SOURCES:$(SRCDIR)/%.cu=$(BUILDDIR)/%.o)
TARGET = $(BINDIR)/hello_cuda

# Section 4 소스 파일들
SECTION4_SOURCES = $(wildcard $(SRCDIR)/Section\ 4/*.cu)
SECTION4_TARGETS = $(SECTION4_SOURCES:$(SRCDIR)/Section\ 4/%.cu=$(BINDIR)/%)

# Section 5 소스 파일들
SECTION5_SOURCES = $(wildcard $(SRCDIR)/Section\ 5/*.cu)
SECTION5_TARGETS = $(SECTION5_SOURCES:$(SRCDIR)/Section\ 5/%.cu=$(BINDIR)/%)

all: $(TARGET) $(SECTION4_TARGETS) $(SECTION5_TARGETS)

$(TARGET): $(OBJECTS) | $(BINDIR)
	$(NVCC) $(CUDA_FLAGS) $(OBJECTS) -o $@ $(LIBS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.cu | $(BUILDDIR)
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) -c $< -o $@

# Section 4 타겟들 (특정 이름으로 명시)
$(BINDIR)/bank_conflict: $(SRCDIR)/Section\ 4/bank_conflict.cu | $(BINDIR)
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) "$(SRCDIR)/Section 4/bank_conflict.cu" -o $@ $(LIBS)

$(BINDIR)/pipeline: $(SRCDIR)/Section\ 4/pipeline.cu | $(BINDIR)
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) "$(SRCDIR)/Section 4/pipeline.cu" -o $@ $(LIBS)

$(BINDIR)/async: $(SRCDIR)/Section\ 4/async.cu | $(BINDIR)
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) "$(SRCDIR)/Section 4/async.cu" -o $@ $(LIBS)

$(BINDIR)/const_memory: $(SRCDIR)/Section\ 4/const_memory.cu | $(BINDIR)
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) "$(SRCDIR)/Section 4/const_memory.cu" -o $@ $(LIBS)

$(BINDIR)/texture_memory: $(SRCDIR)/Section\ 4/texture_memory.cu | $(BINDIR)
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) "$(SRCDIR)/Section 4/texture_memory.cu" -o $@ $(LIBS)

$(BINDIR)/zero_copy: $(SRCDIR)/Section\ 4/zero_copy.cu | $(BINDIR)
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) "$(SRCDIR)/Section 4/zero_copy.cu" -o $@ $(LIBS)

# Section 5 타겟들
$(BINDIR)/multi_gpu: $(SRCDIR)/Section\ 5/multi_gpu.cu | $(BINDIR)
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) "$(SRCDIR)/Section 5/multi_gpu.cu" -o $@ $(LIBS)

$(BINDIR)/gpu_info: $(SRCDIR)/Section\ 5/gpu_info.cu | $(BINDIR)
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) "$(SRCDIR)/Section 5/gpu_info.cu" -o $@ $(LIBS)

$(BINDIR)/gpu_communication: $(SRCDIR)/Section\ 5/gpu_communication.cu | $(BINDIR)
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) "$(SRCDIR)/Section 5/gpu_communication.cu" -o $@ $(LIBS)

$(BINDIR)/gpu_curand: $(SRCDIR)/Section\ 5/gpu_curand.cu | $(BINDIR)
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) "$(SRCDIR)/Section 5/gpu_curand.cu" -o $@ $(LIBS) -lcurand

$(BINDIR)/gpu_fibonacci: $(SRCDIR)/Section\ 5/gpu_fibonacci.cu | $(BINDIR)
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) "$(SRCDIR)/Section 5/gpu_fibonacci.cu" -o $@ $(LIBS)

$(BINDIR)/gpu_fibonacci_parallel: $(SRCDIR)/Section\ 5/gpu_fibonacci_parallel.cu | $(BINDIR)
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) "$(SRCDIR)/Section 5/gpu_fibonacci_parallel.cu" -o $@ $(LIBS)

$(BINDIR)/gpu_factorial: $(SRCDIR)/Section\ 5/gpu_factorial.cu | $(BINDIR)
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) "$(SRCDIR)/Section 5/gpu_factorial.cu" -o $@ $(LIBS)

$(BINDIR)/gpu_factorial_parallel: $(SRCDIR)/Section\ 5/gpu_factorial_parallel.cu | $(BINDIR)
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) "$(SRCDIR)/Section 5/gpu_factorial_parallel.cu" -o $@ $(LIBS)

$(BINDIR)/gpu_predicated_kernel: $(SRCDIR)/Section\ 5/gpu_predicated_kernel.cu | $(BINDIR)
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) "$(SRCDIR)/Section 5/gpu_predicated_kernel.cu" -o $@ $(LIBS)

$(BINDIR):
	mkdir -p $(BINDIR)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

clean:
	rm -rf $(BUILDDIR) $(BINDIR)

run: $(TARGET)
	./$(TARGET)

# Section 4 개별 실행 타겟들
run-bank-conflict: $(BINDIR)/bank_conflict
	./$(BINDIR)/bank_conflict

run-pipeline: $(BINDIR)/pipeline
	./$(BINDIR)/pipeline

# Section 5 개별 실행 타겟들
run-multi-gpu: $(BINDIR)/multi_gpu
	./$(BINDIR)/multi_gpu

run-gpu-info: $(BINDIR)/gpu_info
	./$(BINDIR)/gpu_info

run-gpu-communication: $(BINDIR)/gpu_communication
	./$(BINDIR)/gpu_communication

run-gpu-curand: $(BINDIR)/gpu_curand
	./$(BINDIR)/gpu_curand

run-gpu-fibonacci: $(BINDIR)/gpu_fibonacci
	./$(BINDIR)/gpu_fibonacci

run-gpu-fibonacci-parallel: $(BINDIR)/gpu_fibonacci_parallel
	./$(BINDIR)/gpu_fibonacci_parallel

run-gpu-factorial: $(BINDIR)/gpu_factorial
	./$(BINDIR)/gpu_factorial

run-gpu-factorial-parallel: $(BINDIR)/gpu_factorial_parallel
	./$(BINDIR)/gpu_factorial_parallel

run-gpu-predicated-kernel: $(BINDIR)/gpu_predicated_kernel
	./$(BINDIR)/gpu_predicated_kernel

.PHONY: all clean run run-bank-conflict run-pipeline run-multi-gpu run-gpu-info run-gpu-communication run-gpu-curand run-gpu-fibonacci run-gpu-fibonacci-parallel run-gpu-factorial run-gpu-factorial-parallel run-gpu-predicated-kernel

.PHONY: all clean run run-bank-conflict
