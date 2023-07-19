ARCH_FLAGS = -arch=sm_86
#DEBUG_FLAGS = -g -G
DEBUG_FLAGS = -O3

SRC_DIR = src
BUILD_DIR = build

SRCS = $(SRC_DIR)/main.cu
INCS = $(wildcard $(SRC_DIR)/*.h) $(wildcard $(SRC_DIR)/*.cuh)


.PHONY: all
all: $(BUILD_DIR)/cuda_rt

$(BUILD_DIR)/cuda_rt: $(BUILD_DIR)/cuda_rt.o
	@mkdir -p $(BUILD_DIR)
	nvcc $(ARCH_FLAGS) $(DEBUG_FLAGS) -o $@ $<

$(BUILD_DIR)/cuda_rt.o: $(SRCS) $(INCS)
	@mkdir -p $(BUILD_DIR)
	nvcc $(ARCH_FLAGS) $(DEBUG_FLAGS) -o $@ -c $<

.PHONY: clean
clean:
	rm -rf build

.PHONY: run
run: $(BUILD_DIR)/cuda_rt
	./$(BUILD_DIR)/cuda_rt $(ARGS)
	