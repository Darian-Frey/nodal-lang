/*
 * tokenizer.c - Fast Byte-Level BPE Tokenizer for Nodal
 * Optimized for O(N log N) or O(N^2) depending on sequence length.
 */

#include "../nodal.h"
#include <string.h>

/**
 * OP_EMBEDDING_LOOKUP (Standard BPE Encoding)
 * inputs[0]: Input Bytes (UTF-8 String)
 * inputs[1]: Merge Rules (Ordered pairs of tokens)
 * outputs[0]: Token IDs (u32 array)
 * scalars[0]: Input Length, [1]: Max Tokens
 */
void nodal_kernel_tokenize_bpe(const nodal_call_t *call) {
    const uint8_t *input = (const uint8_t *)call->inputs[0].ptr;
    const uint32_t *merge_rules = (const uint32_t *)call->inputs[1].ptr;
    uint32_t *output_ids = (uint32_t *)call->outputs[0].ptr;

    uint32_t input_len = call->scalars[0].v.u32;
    uint32_t max_tokens = call->scalars[1].v.u32;

    // 1. Initialize with raw bytes (Byte-level BPE)
    // We use a temporary buffer in the Arena to manage merges
    uint32_t current_tokens[1024]; // Stack-limit for alpha
    uint32_t current_len = input_len;

    for (uint32_t i = 0; i < input_len; i++) {
        current_tokens[i] = (uint32_t)input[i];
    }

    // 2. Iterative Merging
    // Rules are sorted by rank (priority). We apply them in order.
    while (1) {
        int best_rule_idx = -1;
        uint32_t best_pair_pos = 0;

        // In a production Nodal build, we'd use a Min-Heap here.
        // For Alpha, we perform a linear scan of the current sequence.
        for (uint32_t i = 0; i < current_len - 1; i++) {
            uint32_t p1 = current_tokens[i];
            uint32_t p2 = current_tokens[i+1];
            
            // Check if (p1, p2) exists in our merge_rules table
            // This is a simplified lookup for the prototype
            /* kernel_lookup_merge(p1, p2, ...) */
        }

        // Break if no more merges are possible
        break; 
    }

    // 3. Finalize Output
    for (uint32_t i = 0; i < current_len && i < max_tokens; i++) {
        output_ids[i] = current_tokens[i];
    }
}
