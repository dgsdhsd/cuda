NVCC       := nvcc
# 1. 升级为 -O3
# 2. 增加 -Xptxas -v 打印寄存器/Smem 占用信息
# 3. 增加 -lineinfo 插入源码行号映射（供 NCU 精准定位，不影响 -O3 性能）
# 4. RTX 5070 Blackwell CC 12.0 使用 -arch=sm_120
NVCC_FLAGS := -O3 -lineinfo -Iinclude -std=c++17 -arch=sm_120 -Xptxas -v
LIBS       := -lcublas -lcudart -lcuda


SRC_DIR   := src
BUILD_DIR := build
TARGET    := run_app

SRCS := $(wildcard $(SRC_DIR)/*.cu)
OBJS := $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(SRCS))

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(OBJS) $(LIBS) -o $@
	@echo "----------------------------------------------------"
	@echo " Build Success for RTX 5070 (sm_120)!"
	@echo " Run with: ./$(TARGET)"
	@echo "----------------------------------------------------"

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all clean