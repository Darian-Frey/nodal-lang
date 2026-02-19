/* * nodal.h - V1.0 Core Runtime Definitions
 * Part of the Nodal Programming Language
 */

#ifndef NODAL_CORE_H
#define NODAL_CORE_H

#include <stdint.h>
#include <stddef.h>

/* --- Constants & Limits --- */
#define NODAL_MAX_INPUTS   6
#define NODAL_MAX_OUTPUTS  4
#define NODAL_MAX_SCALARS  8
#define NODAL_MAX_RANK     4

/* --- Byte-Level Types --- */
typedef uint32_t TensorId;
typedef uint32_t OpId;

typedef enum {
    NDBN_DTYPE_F32  = 0,
    NDBN_DTYPE_F16  = 1,
    NDBN_DTYPE_BF16 = 2,
    NDBN_DTYPE_I8   = 3,
    NDBN_DTYPE_NF4  = 4
} nodal_dtype_t;

/* --- ABI Structures --- */

typedef struct {
    void* ptr;
    size_t byte_len;
} nodal_buffer_t;

typedef struct {
    enum { NODAL_U32, NODAL_I32, NODAL_F32 } kind;
    union {
        uint32_t u32;
        int32_t  i32;
        float    f32;
    } v;
} nodal_scalar_t;

/* The primary contract between Executor and Micro-Kernel */
typedef struct {
    uint16_t       impl_id;
    uint8_t        num_inputs;
    uint8_t        num_outputs;
    uint8_t        num_scalars;
    nodal_buffer_t inputs[NODAL_MAX_INPUTS];
    nodal_buffer_t outputs[NODAL_MAX_OUTPUTS];
    nodal_scalar_t scalars[NODAL_MAX_SCALARS];
} nodal_call_t;

/* --- IR Definitions --- */

typedef enum {
    OP_MATMUL,
    OP_MATMUL_QNF4,
    OP_EMBEDDING_LOOKUP,
    OP_SOFTMAX,
    OP_RMS_NORM,
    OP_ADD
} nodal_op_kind_t;

typedef struct {
    nodal_op_kind_t kind;
    TensorId        inputs[NODAL_MAX_INPUTS];
    TensorId        outputs[NODAL_MAX_OUTPUTS];
    uint8_t         num_inputs;
    uint8_t         num_outputs;
    /* In a real build, scalar_params are often stored in a side-table */
} nodal_irop_t;

#endif /* NODAL_CORE_H */
