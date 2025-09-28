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

all: $(TARGET)

$(TARGET): $(OBJECTS) | $(BINDIR)
	$(NVCC) $(CUDA_FLAGS) $(OBJECTS) -o $@ $(LIBS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.cu | $(BUILDDIR)
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) -c $< -o $@

$(BINDIR):
	mkdir -p $(BINDIR)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

clean:
	rm -rf $(BUILDDIR) $(BINDIR)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
