/*
 * loader.c - Nodal Binary (.nbbin) Loader
 * Implements zero-copy memory mapping for weights and aux data.
 */

#include "nodal.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

/**
 * nodal_load_model_mapped
 * Maps an .nbbin file and populates the runtime buffer table.
 * * @param path: Path to the .nbbin file.
 * @param out_runtime: Array of nodal_buffer_t to be populated.
 * @param max_tensors: Capacity of the out_runtime array.
 * @return: Pointer to the mmap'd base address (needed for munmap later).
 */
void* nodal_load_model_mapped(const char *path, 
                             nodal_buffer_t *out_runtime, 
                             uint32_t max_tensors) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror("Nodal Loader: Failed to open model file");
        return NULL;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return NULL;
    }

    // Perform the memory map (MAP_PRIVATE for read-only efficiency)
    void *base = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd); // File descriptor no longer needed after mmap

    if (base == MAP_FAILED) {
        perror("Nodal Loader: mmap failed");
        return NULL;
    }

    uint8_t *u8_base = (uint8_t *)base;

    /* 1. Validate Header (Simple offset check for V1) */
    uint32_t magic = *(uint32_t*)u8_base;
    if (magic != 0x4E42444E) { // 'NDBN'
        fprintf(stderr, "Nodal Loader: Invalid Magic Number\n");
        munmap(base, st.st_size);
        return NULL;
    }

    /* 2. Locate Tensor Table
     * Based on our spec: Table offset is at byte 12 (0x0C)
     */
    uint64_t table_offset = *(uint64_t*)(u8_base + 12);
    uint32_t tensor_count = *(uint32_t*)(u8_base + 8);

    if (tensor_count > max_tensors) {
        fprintf(stderr, "Nodal Loader: Buffer overflow protection triggered\n");
        munmap(base, st.st_size);
        return NULL;
    }

    /* 3. Populate Runtime Buffer Table
     * Each TensorEntry is 64 bytes. We extract the data_offset and data_size.
     */
    uint8_t *table_ptr = u8_base + table_offset;
    for (uint32_t i = 0; i < tensor_count; ++i) {
        uint8_t *entry = table_ptr + (i * 64);
        
        uint64_t data_offset = *(uint64_t*)(entry + 24); // Based on NDBN_TensorEntry
        uint64_t data_size   = *(uint64_t*)(entry + 32);

        out_runtime[i].ptr = (void *)(u8_base + data_offset);
        out_runtime[i].byte_len = (size_t)data_size;
    }

    return base;
}
