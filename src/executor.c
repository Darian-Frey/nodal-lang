/*
 * executor.c - The Tiny Dispatch Loop for Nodal V1.0
 * Performs zero-allocation dispatch from IR to Micro-Kernels.
 */

#include "nodal.h"
#include <stddef.h>

/* Forward declaration of the kernel function pointer type */
typedef void (*nodal_kernel_fn)(const nodal_call_t *call);

/* Internal table entry for mapping Ops to Kernels */
typedef struct {
    nodal_op_kind_t kind;
    nodal_kernel_fn fn;
} nodal_kernel_mapping_t;

/* * The Kernel Registry (Static Table)
 * In a full build, this would be populated based on the target (ARM, RISC-V, etc.)
 */
static const nodal_kernel_mapping_t KERNEL_REGISTRY[] = {
    /* {OP_MATMUL_QNF4, nodal_kernel_matmul_qnf4_arm}, */
    /* {OP_EMBEDDING_LOOKUP, nodal_kernel_embed_generic}, */
};

static const size_t KERNEL_REGISTRY_COUNT = sizeof(KERNEL_REGISTRY) / sizeof(KERNEL_REGISTRY[0]);

/**
 * nodal_execute_tape
 * Dispatches a sequence of IROps.
 * @param ops: The linear tape of operations.
 * @param op_count: Number of operations in the tape.
 * @param tensor_runtime: Resolved pointers for every TensorID.
 */
void nodal_execute_tape(const nodal_irop_t *ops, 
                        size_t op_count, 
                        const nodal_buffer_t *tensor_runtime) {
    
    // Stack-allocated call descriptor (Zero-Allocation)
    nodal_call_t call;

    for (size_t i = 0; i < op_count; ++i) {
        const nodal_irop_t *op = &ops[i];
        nodal_kernel_fn kernel = NULL;

        // 1. Resolve Kernel Function from Registry
        for (size_t k = 0; k < KERNEL_REGISTRY_COUNT; ++k) {
            if (KERNEL_REGISTRY[k].kind == op->kind) {
                kernel = KERNEL_REGISTRY[k].fn;
                break;
            }
        }

        if (!kernel) continue; // Safety: Skip unknown ops (or trigger RECOVERY_NODE)

        // 2. Resolve TensorID to Physical Pointers
        call.num_inputs = op->num_inputs;
        for (uint8_t j = 0; j < op->num_inputs; ++j) {
            call.inputs[j] = tensor_runtime[op->inputs[j]];
        }

        call.num_outputs = op->num_outputs;
        for (uint8_t j = 0; j < op->num_outputs; ++j) {
            call.outputs[j] = tensor_runtime[op->outputs[j]];
        }

        // 3. Dispatch to Micro-Kernel
        kernel(&call);
    }
}
