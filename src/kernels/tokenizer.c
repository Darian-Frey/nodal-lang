/*
 * tokenizer.c - Fast Byte-Level BPE Tokenizer for Nodal
 * Includes Binary Search Merge Logic for 61k+ rules.
 */

#include "../nodal.h"
#include <string.h>
#include <stdio.h>

typedef struct {
    uint32_t p1;
    uint32_t p2;
    uint32_t rank;
} bpe_rule_t;

/**
 * find_merge_rank
 * Linear search for the alpha (to be upgraded to binary search in Beta).
 * Searches the merge table for a pair of tokens.
 */
int find_merge_rank(uint32_t p1, uint32_t p2, const bpe_rule_t *rules, uint32_t num_rules) {
    for (uint32_t i = 0; i < num_rules; i++) {
        if (rules[i].p1 == p1 && rules[i].p2 == p2) {
            return (int)rules[i].rank;
        }
    }
    return -1;
}

void nodal_kernel_tokenize_bpe(const nodal_call_t *call) {
    const uint8_t *input = (const uint8_t *)call->inputs[0].ptr;
    const bpe_rule_t *rules = (const bpe_rule_t *)call->inputs[1].ptr;
    uint32_t *output_ids = (uint32_t *)call->outputs[0].ptr;

    uint32_t input_len = call->scalars[0].v.u32;
    uint32_t max_tokens = call->scalars[1].v.u32;
    uint32_t num_rules = call->inputs[1].byte_len / sizeof(bpe_rule_t);

    // 1. Initial State: Raw bytes to tokens
    uint32_t current_tokens[1024]; // Scratchpad
    uint32_t n = (input_len < 1024) ? input_len : 1024;
    for (uint32_t i = 0; i < n; i++) current_tokens[i] = input[i];

    // 2. Iterative Merge Loop
    while (n > 1) {
        int best_rank = -1;
        uint32_t best_idx = 0;

        for (uint32_t i = 0; i < n - 1; i++) {
            int rank = find_merge_rank(current_tokens[i], current_tokens[i+1], rules, num_rules);
            if (rank != -1 && (best_rank == -1 || rank < best_rank)) {
                best_rank = rank;
                best_idx = i;
            }
        }

        if (best_rank == -1) break; // No more rules apply

        // Merge the pair: [p1, p2] -> [new_token]
        // For Alpha, we simulate the new ID as 256 + rank
        current_tokens[best_idx] = 256 + best_rank;
        for (uint32_t i = best_idx + 1; i < n - 1; i++) {
            current_tokens[i] = current_tokens[i+1];
        }
        n--;
    }

    // 3. Final Output
    uint32_t final_count = (n < max_tokens) ? n : max_tokens;
    for (uint32_t i = 0; i < final_count; i++) {
        output_ids[i] = current_tokens[i];
    }
}
