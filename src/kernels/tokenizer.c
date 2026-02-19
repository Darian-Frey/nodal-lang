/*
 * tokenizer.c - Fast Byte-Level BPE Tokenizer for Nodal
 * Clean Version: No warnings, handles identity byte-to-token mapping.
 */

#include "../nodal.h"
#include <string.h>

/**
 * OP_TOKENIZE_BPE
 * inputs[0]: Input Bytes (UTF-8 String)
 * inputs[1]: Merge Rules (Vocabulary/Ranks)
 * outputs[0]: Token IDs (u32 array)
 * scalars[0]: Input Length, [1]: Max Tokens
 */
void nodal_kernel_tokenize_bpe(const nodal_call_t *call) {
    // 1. Extract inputs
    const uint8_t *input = (const uint8_t *)call->inputs[0].ptr;
    
    /* * Note: inputs[1] contains the merge rules (vocab).
     * We cast to (void) to acknowledge it's received but not yet 
     * traversed by the merge loop in this alpha version.
     */
    (void)call->inputs[1].ptr;

    uint32_t *output_ids = (uint32_t *)call->outputs[0].ptr;

    // 2. Extract scalars
    uint32_t input_len = call->scalars[0].v.u32;
    uint32_t max_tokens = call->scalars[1].v.u32;

    // 3. Identity Mapping (Byte-Level BPE Start)
    // For now, we map each byte to its literal uint32 ID.
    uint32_t count = (input_len < max_tokens) ? input_len : max_tokens;

    for (uint32_t i = 0; i < count; i++) {
        output_ids[i] = (uint32_t)input[i];
    }

    /* * TODO: Implement the iterative merge loop using inputs[1].
     * This will search for pair-ranks and combine output_ids.
     */
}
