NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_80 # 请根据你的 GPU 架构修改 sm_xx，比如 3090 是 sm_86，4090 是 sm_89

INCLUDES = -I./include
SRC = src/memory_pool.cu src/transformer_layer.cu main.cu
OBJ = $(SRC:.cu=.o)
TARGET = gemm_simulator

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $^

clean:
	rm -f $(TARGET)
