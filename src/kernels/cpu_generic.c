/*
 * cpu_generic.c - Reference implementations for Nodal Ops
 * Use these for validation and as fallbacks for non-accelerated hardware.
 */

#include "../nodal.h"
#include <math.h>

/**
 * OP_MATMUL (Generic F32)
 * C = A * B
 * scalars[0]=M, [1]=N, [2]=K
 */
void nodal_kernel_matmul_generic(const nodal_call_t *call) {
    const float *A = (const float *)call->inputs[0].ptr;
    const float *B = (const float *)call->inputs[1].ptr;
    float *C = (float *)call->outputs[0].ptr;

    uint32_t M = call->scalars[0].v.u32;
    uint32_t N = call->scalars[1].v.u32;
    uint32_t K = call->scalars[2].v.u32;

    for (uint32_t i = 0; i < M; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/**
 * OP_SOFTMAX (Generic F32)
 * scalars[0]=size
 */
void nodal_kernel_softmax_generic(const nodal_call_t *call) {
    const float *in = (const float *)call->inputs[0].ptr;
    float *out = (float *)call->outputs[0].ptr;
    uint32_t size = call->scalars[0].v.u32;

    float max_val = in[0];
    for (uint32_t i = 1; i < size; ++i) {
        if (in[i] > max_val) max_val = in[i];
    }

    float sum = 0.0f;
    for (uint32_t i = 0; i < size; ++i) {
        out[i] = expf(in[i] - max_val);
        sum += out[i];
    }

    for (uint32_t i = 0; i < size; ++i) {
        out[i] /= sum;
    }
}

/**
 * OP_ADD (Element-wise F32)
 * scalars[0]=size
 */
void nodal_kernel_add_generic(const nodal_call_t *call) {
    const float *A = (const float *)call->inputs[0].ptr;
    const float *B = (const float *)call->inputs[1].ptr;
    float *C = (float *)call->outputs[0].ptr;
    uint32_t size = call->scalars[0].v.u32;

    for (uint32_t i = 0; i < size; ++i) {
        C[i] = A[i] + B[i];
    }
}
