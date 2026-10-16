# Parallel search in a binary search tree - CPU (pthreads) and GPU (CUDA) versions.
#
#   make                         build the three CPU versions
#   make gpu                     build all the GPU versions (needs nvcc)
#   make SEED=7 NODES=50000 QUERIES=100000   build with a different / smaller dataset
#   make clean
#
# The original measurements were taken with these default flags: no
# optimization for the CPU code (gcc default, -O0) and device debug info (-G)
# for the CUDA code. Override CFLAGS / NVCCFLAGS to experiment.

CC        ?= gcc
CFLAGS    ?= -std=gnu11 -Wall
LDLIBS    := -lpthread -lm
NVCC      ?= nvcc
NVCCFLAGS ?= -arch=sm_75 -G
PYTHON    ?= python3

# Dataset parameters (see scripts/generate_dataset.py)
SEED      ?= 1
NODES     ?= 500000
QUERIES   ?= 10000000
MAX_VALUE ?= 4000000000

BUILD_DIR := build
BIN_DIR   := bin
DATA_DIR  := $(BUILD_DIR)/data/seed$(SEED)_n$(NODES)_q$(QUERIES)
DATASET_C := $(DATA_DIR)/dataset.c
DATASET_O := $(DATA_DIR)/dataset.o

# Suffix appended to the executables, e.g. BIN_SUFFIX=_07 -> bin/cpu_v3_07
BIN_SUFFIX ?=

INCLUDES := -Iinclude
COMMON   := src/common/bintree.c

CPU_V1_SRC := src/cpu/v1_dispatcher/main.c src/cpu/v1_dispatcher/search_v1.c
CPU_V2_SRC := src/cpu/v2_range_split/main.c src/cpu/v2_range_split/search_v2.c src/cpu/v2_range_split/range_utils.c
CPU_V3_SRC := src/cpu/v3_equal_split/main.c src/cpu/v3_equal_split/search_v3.c

GPU_VERSIONS := v1 v2_1 v2_2 v3
GPU_SRC_v1    := src/gpu/v1_pointer_tree.cu
GPU_SRC_v2_1  := src/gpu/v2_1_indexed_array.cu
GPU_SRC_v2_2  := src/gpu/v2_2_implicit_array.cu
GPU_SRC_v3    := src/gpu/v3_shared_memory.cu

CPU_BINS := $(BIN_DIR)/cpu_v1$(BIN_SUFFIX) $(BIN_DIR)/cpu_v2$(BIN_SUFFIX) $(BIN_DIR)/cpu_v3$(BIN_SUFFIX)
GPU_BINS := $(foreach v,$(GPU_VERSIONS),$(BIN_DIR)/gpu_$(v)$(BIN_SUFFIX))

.PHONY: all cpu gpu dataset clean distclean
.SECONDARY: $(DATASET_C)

all: cpu

cpu: $(CPU_BINS)

gpu: $(GPU_BINS)

dataset: $(DATASET_C)

$(DATASET_C): scripts/generate_dataset.py
	$(PYTHON) scripts/generate_dataset.py --seed $(SEED) --nodes $(NODES) --queries $(QUERIES) --max-value $(MAX_VALUE) --out $@

$(DATASET_O): $(DATASET_C) include/dataset.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(BIN_DIR):
	mkdir -p $@

# ---- CPU ----------------------------------------------------------------------

$(BIN_DIR)/cpu_v1$(BIN_SUFFIX): $(CPU_V1_SRC) $(COMMON) include/bintree.h | $(BIN_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(CPU_V1_SRC) $(COMMON) -o $@ $(LDLIBS)

$(BIN_DIR)/cpu_v2$(BIN_SUFFIX): $(CPU_V2_SRC) $(COMMON) $(DATASET_O) include/bintree.h | $(BIN_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(CPU_V2_SRC) $(COMMON) $(DATASET_O) -o $@ $(LDLIBS)

$(BIN_DIR)/cpu_v3$(BIN_SUFFIX): $(CPU_V3_SRC) $(COMMON) $(DATASET_O) include/bintree.h | $(BIN_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(CPU_V3_SRC) $(COMMON) $(DATASET_O) -o $@ $(LDLIBS)

# ---- GPU ----------------------------------------------------------------------

define GPU_RULE
$(BIN_DIR)/gpu_$(1)$(BIN_SUFFIX): $(GPU_SRC_$(1)) src/gpu/main.c src/gpu/search_cuda.h src/gpu/gpu_utils.cuh $(COMMON) $(DATASET_O) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -Isrc/gpu $(COMMON) $(DATASET_O) src/gpu/main.c $(GPU_SRC_$(1)) -o $$@
endef
$(foreach v,$(GPU_VERSIONS),$(eval $(call GPU_RULE,$(v))))

# -------------------------------------------------------------------------------

clean:
	rm -rf $(BIN_DIR)

distclean: clean
	rm -rf $(BUILD_DIR)
