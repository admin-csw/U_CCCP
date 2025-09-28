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

all: $(TARGET) $(SECTION4_TARGETS)

$(TARGET): $(OBJECTS) | $(BINDIR)
	$(NVCC) $(CUDA_FLAGS) $(OBJECTS) -o $@ $(LIBS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.cu | $(BUILDDIR)
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) -c $< -o $@

# Section 4 타겟들
$(BINDIR)/%: $(SRCDIR)/Section\ 4/%.cu | $(BINDIR)
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) "$(SRCDIR)/Section 4/$*.cu" -o $@ $(LIBS)

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

.PHONY: all clean run run-bank-conflict run-pipeline

.PHONY: all clean run run-bank-conflict
