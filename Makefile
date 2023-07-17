ARCH_FLAGS = -arch=sm_86

SRC_DIR = src
BUILD_DIR = build

SRCS = $(SRC_DIR)/main.cu
INCS = $(SRC_DIR)/render_option.h


.PHONY: all
all: $(BUILD_DIR)/cuda_rt

$(BUILD_DIR)/cuda_rt: $(BUILD_DIR)/cuda_rt.o
	@mkdir -p $(BUILD_DIR)
	nvcc $(ARCH_FLAGS) -o $@ $^

$(BUILD_DIR)/cuda_rt.o: $(SRCS) $(INCS)
	@mkdir -p $(BUILD_DIR)
	nvcc $(ARCH_FLAGS) -o $@ -dc $<

.PHONY: clean
clean:
	rm -rf build

.PHONY: run
run: $(BUILD_DIR)/cuda_rt
	./$(BUILD_DIR)/cuda_rt $(ARGS)