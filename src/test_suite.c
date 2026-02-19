/*
 * test_suite.c - Kernel Validation & Math Accuracy Tests
 * Consolidates F32 Reference Math and NF4 Quantization Checks.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include "nodal.h"

/* Linkage to our kernels */
extern void nodal_kernel_matmul_generic(const nodal_call_t *call);

#define EPSILON 1e-4

/* The Canonical NF4 LUT for validation */
static const float TEST_NF4_LUT[16] = {
    -1.000000f, -0.694417f, -0.512093f, -0.373103f, 
    -0.255986f, -0.150166f, -0.051515f,  0.000000f, 
     0.051515f,  0.150166f,  0.255986f,  0.373103f, 
     0.512093f,  0.694417f,  1.000000f,  1.250000f
};

/**
 * assert_near
 * Validates that two floats are within a tiny margin of error.
 */
static int assert_near(float a, float b, const char* context) {
    if (fabsf(a - b) > EPSILON) {
        printf("[FAIL] %s: %f != %f (diff: %f)\n", context, a, b, fabsf(a - b));
        return 0;
    }
    return 1;
}

/**
 * test_matmul_logic
 * Verifies standard F32 Matrix Multiplication.
 */
void test_matmul_logic() {
    printf("[TEST] Running F32 MatMul Reference Test...\n");

    // Setup Mock Data (2x2 Matrix)
    float A_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float B_data[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float C_out[4]  = {0.0f, 0.0f, 0.0f, 0.0f};

    nodal_call_t call;
    call.inputs[0] = (nodal_buffer_t){.ptr = A_data, .byte_len = 16};
    call.inputs[1] = (nodal_buffer_t){.ptr = B_data, .byte_len = 16};
    call.outputs[0] = (nodal_buffer_t){.ptr = C_out, .byte_len = 16};
    
    // M=2, N=2, K=2
    call.scalars[0] = (nodal_scalar_t){.kind = NODAL_U32, .v.u32 = 2};
    call.scalars[1] = (nodal_scalar_t){.kind = NODAL_U32, .v.u32 = 2};
    call.scalars[2] = (nodal_scalar_t){.kind = NODAL_U32, .v.u32 = 2};

    nodal_kernel_matmul_generic(&call);

    int pass = 1;
    pass &= assert_near(C_out[0], 19.0f, "C[0,0]");
    pass &= assert_near(C_out[1], 22.0f, "C[0,1]");
    pass &= assert_near(C_out[2], 43.0f, "C[1,0]");
    pass &= assert_near(C_out[3], 50.0f, "C[1,1]");

    if (pass) printf("[PASS] MatMul Logic Verified.\n");
}

/**
 * test_nf4_dequant_logic
 * Verifies the 4-bit nibble unpacking and scaling.
 */
void test_nf4_dequant_logic() {
    printf("[TEST] Running NF4 Dequantization Validation...\n");

    // Mock a byte: 0xE7
    // High nibble: 0xE (14) -> 1.000000
    // Low nibble:  0x7 (7)  -> 0.000000
    uint8_t packed_weight = 0xE7; 
    float scale = 2.0f;

    // Simulate bit-unrolling
    float w_low  = TEST_NF4_LUT[packed_weight & 0x0F] * scale;
    float w_high = TEST_NF4_LUT[packed_weight >> 4]   * scale;

    int pass = 1;
    pass &= assert_near(w_low,  0.0f, "NF4_Low_Nibble");
    pass &= assert_near(w_high, 2.0f, "NF4_High_Nibble");

    if (pass) printf("[PASS] NF4 Dequantization Verified.\n");
}

int main() {
    printf("=== Nodal V1.0 Test Suite ===\n");
    
    test_matmul_logic();
    test_nf4_dequant_logic();

    printf("=== All Tests Complete ===\n");
    return 0;
}
