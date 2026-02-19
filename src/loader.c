/*
 * loader.c - Zero-Copy Model Loader for Nodal
 * Maps .nbbin files and resolves Tensor/Vocab pointers.
 */

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include "nodal.h"

/**
 * nodal_load_model_mapped
 * Maps a .nbbin file into virtual memory and populates the tensor runtime.
 * * @param path         Path to the .nbbin file.
 * @param out_runtime  Pointer to an array of nodal_buffer_t to be populated.
 * @param max_tensors  Size of the out_runtime array.
 * @return             The base address of the mapping (for future munmap).
 */
void* nodal_load_model_mapped(const char *path, nodal_buffer_t *out_runtime, uint32_t max_tensors) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror("[LOADER] Error opening file");
        return NULL;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return NULL;
    }
    size_t size = st.st_size;

    // 1. Memory Map the entire file (Read-Only, Private)
    void *base = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd); // FD no longer needed after mapping

    if (base == MAP_FAILED) {
        perror("[LOADER] mmap failed");
        return NULL;
    }

    // 2. Validate Header
    nodal_header_t *hdr = (nodal_header_t *)base;
    if (hdr->magic != 0x4E42444E) { // 'NDBN'
        fprintf(stderr, "[LOADER] Invalid Magic Number: 0x%08X\n", hdr->magic);
        munmap(base, size);
        return NULL;
    }

    printf("[LOADER] Mapping Model v%d (%u tensors)\n", hdr->version, hdr->num_tensors);

    // 3. Resolve Vocabulary (String Table)
    // We map the vocab to the last slot in the runtime table as a convention.
    if (hdr->string_table_offset > 0 && hdr->string_table_offset < size) {
        uint32_t vocab_idx = max_tensors - 1;
        out_runtime[vocab_idx].ptr = (uint8_t*)base + hdr->string_table_offset;
        out_runtime[vocab_idx].byte_len = size - hdr->string_table_offset;
        printf("[LOADER] Vocab segment mapped to runtime index [%u]\n", vocab_idx);
    }

    // 4. Map Individual Tensors
    // The tensor table starts at hdr->tensor_table_offset
    nodal_tensor_entry_t *table = (nodal_tensor_entry_t *)((uint8_t *)base + hdr->tensor_table_offset);
    
    for (uint32_t i = 0; i < hdr->num_tensors && i < (max_tensors - 1); i++) {
        // Map the main data pointer
        out_runtime[i].ptr = (uint8_t *)base + table[i].data_offset;
        out_runtime[i].byte_len = table[i].data_size;

        /* Note: In a production build, aux_data (scales/min-max) 
         * would be handled by a secondary mapping or an aux_runtime array.
         * For Alpha, we focus on the primary weight data.
         */
    }

    return base;
}
