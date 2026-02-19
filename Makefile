# Nodal V1.0 - Multi-Backend Makefile
# Usage: 
#   make TARGET=generic    (Default: Generic C kernels)
#   make TARGET=arm        (Optimized for ARM Neon)
#   make TARGET=riscv      (Future: RISC-V Vector)

CC = gcc
CFLAGS = -O3 -Wall -Wextra -I./src
LDFLAGS = -lm

# Target Configuration
TARGET ?= generic

# Source Files
SRCS = src/main.c \
       src/executor.c \
       src/loader.c \
       src/kernels/cpu_generic.c

# Backend Specific Settings
ifeq ($(TARGET), arm)
    SRCS += src/kernels/arm_neon_nf4.c
    CFLAGS += -mfpu=neon -march=armv8-a -DNODAL_TARGET_ARM
    BINARY = nodal_arm
else ifeq ($(TARGET), riscv)
    # Placeholder for RISC-V extensions
    CFLAGS += -march=rv64gcv -DNODAL_TARGET_RISCV
    BINARY = nodal_riscv
else
    CFLAGS += -DNODAL_TARGET_GENERIC
    BINARY = nodal_generic
endif

# Build Rules
all: $(BINARY)

$(BINARY): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(BINARY) $(LDFLAGS)

clean:
	rm -f nodal_generic nodal_arm nodal_riscv

.PHONY: all clean
