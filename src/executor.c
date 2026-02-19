/*
 * executor.c - Nodal IR Instruction Dispatcher
 * Aligned with nodal.h V1.0
 */

#include <stdio.h>
#include <string.h>
#include "nodal.h"

/* Kernel Forward Declarations */
extern void nodal_kernel_matmul_generic(const nodal_call_t *call);
extern void nodal_kernel_softmax_generic(const nodal_call_t *call);
extern void nodal_kernel_add_generic(const nodal_call_t *call);
extern void nodal_kernel_tokenize_bpe(const nodal_call_t *call);

/**
 * nodal_execute_tape
 * Iterates through a sequence of IROps and dispatches to kernels.
 */
void nodal_execute_tape(const nodal_irop_t *ops, size_t op_count, const nodal_buffer_t *tensor_runtime) {
    for (size_t i = 0; i < op_count; i++) {
        const nodal_irop_t *op = &ops[i];
        nodal_call_t call;

        // 1. Map IR indices to physical memory pointers
        for (uint32_t j = 0; j < 8; j++) {
            if (j < 8) { // Safety check for fixed array size
                uint32_t in_idx = op->inputs[j];
                call.inputs[j] = tensor_runtime[in_idx];
            }
        }

        for (uint32_t j = 0; j < 4; j++) {
            uint32_t out_idx = op->outputs[j];
            call.outputs[j] = tensor_runtime[out_idx];
        }

        // 2. Copy scalars (parameters like M, N, K)
        memcpy(call.scalars, op->scalars, sizeof(nodal_scalar_t) * 8);

        // 3. Dispatch to the appropriate kernel
        switch (op->kind) {
            case OP_MATMUL:
                nodal_kernel_matmul_generic(&call);
                break;
            case OP_SOFTMAX:
                nodal_kernel_softmax_generic(&call);
                break;
            case OP_ADD:
                nodal_kernel_add_generic(&call);
                break;
            case OP_TOKENIZE_BPE:
                nodal_kernel_tokenize_bpe(&call);
                break;
            default:
                fprintf(stderr, "[EXEC] Unknown OP Code: %d\n", op->kind);
                break;
        }
    }
}
