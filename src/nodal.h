/* * nodal.h - V1.0 Core Runtime Definitions
 * Part of the Nodal Programming Language
 */

#ifndef NODAL_H
#define NODAL_H

#include <stdint.h>
#include <stddef.h>

/* --- Nodal Binary Specification (NDBN) --- */

/**
 * Nodal Header (32 bytes)
 * Fixed-size entry point for the model file.
 */
typedef struct {
    uint32_t magic;                // 0x4E42444E ('NDBN')
    uint16_t version;              // Format version
    uint16_t flags;                // Feature flags
    uint32_t num_tensors;          // Count of tensors in table
    uint32_t tensor_table_offset;  // Offset to start of Tensor Entries
    uint64_t string_table_offset;  // Offset to Vocab/Merge data
    uint64_t reserved;             // Alignment padding
} nodal_header_t;

/**
 * Nodal Tensor Entry (64 bytes)
 * Describes a single tensor's shape, type, and location.
 */
typedef struct {
    uint32_t name_offset;          // Offset to name in string table
    uint8_t  dtype;                // 0=F32, 4=NF4
    uint8_t  rank;                 // Number of dimensions
    uint8_t  layout;               // Memory layout (Row-major)
    uint8_t  has_aux;              // 1 if scale/min-max data exists
    uint32_t shape[4];             // Support for up to 4D tensors
    uint64_t data_offset;          // Offset to raw weights
    uint64_t data_size;            // Size of raw weights in bytes
    uint64_t aux_offset;           // Offset to scales (if has_aux=1)
    uint64_t aux_size;             // Size of scale data
} nodal_tensor_entry_t;

/* --- Runtime Structures --- */

typedef enum {
    NODAL_F32 = 0,
    NODAL_U32 = 1,
    NODAL_NF4 = 4
} nodal_type_t;

/**
 * Nodal Buffer
 * A generic pointer+length descriptor for memory segments.
 */
typedef struct {
    void     *ptr;
    size_t    byte_len;
} nodal_buffer_t;

typedef struct {
    nodal_type_t kind;
    union {
        float    f32;
        uint32_t u32;
    } v;
} nodal_scalar_t;

/**
 * Nodal Call
 * The standard interface for all Nodal Kernels.
 */
typedef struct {
    nodal_buffer_t inputs[8];      // Support up to 8 input tensors
    nodal_buffer_t outputs[4];     // Support up to 4 output tensors
    nodal_scalar_t scalars[8];     // Contextual parameters (M, N, K, etc.)
} nodal_call_t;

/* --- IR Operation Types --- */

typedef enum {
    OP_MATMUL = 0,
    OP_MATMUL_QNF4 = 1,
    OP_SOFTMAX = 2,
    OP_ADD = 3,
    OP_TOKENIZE_BPE = 4
} nodal_op_kind_t;

typedef struct {
    nodal_op_kind_t kind;
    uint32_t num_inputs;
    uint32_t num_outputs;
    uint32_t inputs[8];            // Indices into the runtime tensor table
    uint32_t outputs[4];
    nodal_scalar_t scalars[8];
} nodal_irop_t;

#endif // NODAL_H
