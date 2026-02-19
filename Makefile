# ==========================================
# Nodal V1.0 - Multi-Backend Build System
# ==========================================

CC = gcc
CFLAGS = -O3 -Wall -Wextra -I./src
LDFLAGS = -lm

# Target Configuration (generic, arm, riscv)
TARGET ?= generic

# --- Source Files ---
# Core Runtime Components
CORE_SRCS = src/executor.c src/loader.c src/kernels/cpu_generic.c src/kernels/tokenizer.c

# CLI Entry Point
CLI_SRC = src/cli.c

# Test Suite Entry Point
TEST_SRC = src/test_suite.c

# --- Backend Logic ---
ifeq ($(TARGET), arm)
    CORE_SRCS += src/kernels/arm_neon_nf4.c
    CFLAGS += -mfpu=neon -march=armv8-a -DNODAL_TARGET_ARM
    BINARY = nr_arm
else ifeq ($(TARGET), riscv)
    CFLAGS += -march=rv64gcv -DNODAL_TARGET_RISCV
    BINARY = nr_riscv
else
    CFLAGS += -DNODAL_TARGET_GENERIC
    BINARY = nr
endif

# --- Build Rules ---

.PHONY: all clean test help

all: $(BINARY)

# Main Runtime Binary (nr)
$(BINARY): $(CLI_SRC) $(CORE_SRCS)
	$(CC) $(CFLAGS) $(CLI_SRC) $(CORE_SRCS) -o $(BINARY) $(LDFLAGS)
	@echo "[SUCCESS] Built Nodal Runtime: $(BINARY)"

# Test Suite Binary
test: $(TEST_SRC) $(CORE_SRCS)
	$(CC) $(CFLAGS) $(TEST_SRC) $(CORE_SRCS) -o nodal_test $(LDFLAGS)
	@echo "[TEST] Running Nodal Validation Suite..."
	./nodal_test

# Cleanup
clean:
	rm -f nr nr_arm nr_riscv nodal_test test_model.nbbin
	@echo "[CLEAN] Removed binaries and temporary models."

# Documentation / Help
help:
	@echo "Nodal Build System Commands:"
	@echo "  make              - Build for generic CPU (Default)"
	@echo "  make TARGET=arm   - Build with ARM Neon optimizations"
	@echo "  make test         - Build and run the math validation suite"
	@echo "  make clean        - Remove all generated binaries"
