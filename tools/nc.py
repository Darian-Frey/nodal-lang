import struct
import numpy as np
import argparse
import os
import json

# Canonical NF4 LUT
NF4_LUT = np.array([
    -1.0, -0.6944172978401184, -0.5120928883552551, -0.37310290336608887, 
    -0.25598612427711487, -0.15016591548919678, -0.05151525139808655, 0.0, 
    0.05151525139808655, 0.15016591548919678, 0.25598612427711487, 0.37310290336608887, 
    0.5120928883552551, 0.6944172978401184, 1.0, 1.25
], dtype=np.float32)

def align_to(size, alignment=64):
    return (size + alignment - 1) & ~(alignment - 1)

def quantize_nf4(data, block_size=64):
    flat_data = data.flatten().astype(np.float32)
    num_blocks = (len(flat_data) + block_size - 1) // block_size
    packed_weights = []
    scales = []

    for i in range(num_blocks):
        block = flat_data[i*block_size : (i+1)*block_size]
        if len(block) < block_size:
            block = np.pad(block, (0, block_size - len(block)))
            
        max_abs = np.max(np.abs(block))
        scale = max_abs if max_abs > 0 else 1.0
        scales.append(scale)
        
        indices = np.abs((block / scale)[:, None] - NF4_LUT).argmin(axis=1)
        for j in range(0, block_size, 2):
            low = indices[j] & 0x0F
            high = indices[j+1] & 0x0F
            packed_weights.append((high << 4) | low)

    return bytes(packed_weights), np.array(scales, dtype=np.float32)

class NodalCompiler:
    def __init__(self, output_path):
        self.output_path = output_path
        self.tensors = []
        self.vocab_data = b""

    def load_vocab(self, tokenizer_json_path):
        """Parses tokenizer.json and builds the Nodal merge table."""
        print(f"[VOCAB] Loading {tokenizer_json_path}...")
        with open(tokenizer_json_path, "r") as f:
            data = json.load(f)
        
        # Extract Merges (the rules for BPE)
        merges = data.get("model", {}).get("merges", [])
        binary_merges = b""
        
        # Nodal Binary Vocab Format: [p1_u32][p2_u32][rank_u32]
        # This allows the C kernel to find merges in O(log N) via binary search
        for i, merge_str in enumerate(merges):
            # Typical format: "byte1 byte2"
            parts = merge_str.split()
            if len(parts) == 2:
                # Note: This is a simplified mapping for the Alpha
                # In production, we map strings to their initial byte IDs
                try:
                    p1 = ord(parts[0][0]) if len(parts[0]) == 1 else 0 
                    p2 = ord(parts[1][0]) if len(parts[1]) == 1 else 0
                    binary_merges += struct.pack("<III", p1, p2, i)
                except: continue
        
        self.vocab_data = binary_merges
        print(f"[VOCAB] Compiled {len(merges)} merge rules.")

    def add_tensor(self, name, data, dtype="NF4", block_size=64):
        if dtype == "NF4":
            packed, scales = quantize_nf4(data, block_size)
            aux_header = struct.pack("<BBBBII", 0, 0, 0, 0, block_size, len(scales))
            self.tensors.append({
                "name": name, "data": packed, "aux_data": aux_header + scales.tobytes(),
                "dtype": 4, "shape": list(data.shape)
            })
        else:
            self.tensors.append({
                "name": name, "data": data.tobytes(), "aux_data": None,
                "dtype": 0, "shape": list(data.shape)
            })

    def compile(self):
        with open(self.output_path, "wb") as f:
            f.write(b"\x00" * 32) # Header placeholder
            tensor_table_offset = f.tell()
            for _ in self.tensors: f.write(b"\x00" * 64)

            locs = []
            for t in self.tensors:
                f.write(b"\x00" * (align_to(f.tell()) - f.tell()))
                d_off = f.tell()
                f.write(t["data"])
                d_sz = len(t["data"])
                
                a_off, a_sz = 0, 0
                if t["aux_data"]:
                    f.write(b"\x00" * (align_to(f.tell()) - f.tell()))
                    a_off = f.tell()
                    f.write(t["aux_data"])
                    a_sz = len(t["aux_data"])
                locs.append((d_off, d_sz, a_off, a_sz))

            # Write Vocab at the end
            vocab_offset = 0
            if self.vocab_data:
                f.write(b"\x00" * (align_to(f.tell()) - f.tell()))
                vocab_offset = f.tell()
                f.write(self.vocab_data)

            f.seek(0)
            f.write(struct.pack("<IHHIIQ", 0x4E42444E, 1, 0, len(self.tensors), tensor_table_offset, vocab_offset))
            f.seek(tensor_table_offset)
            for i, t in enumerate(self.tensors):
                d_off, d_sz, a_off, a_sz = locs[i]
                f.write(struct.pack("<IBBBB4IQQQQ", 0, t["dtype"], len(t["shape"]), 0, 1 if t["aux_data"] else 0,
                    *(t["shape"] + [0]*(4-len(t["shape"]))), d_off, d_sz, a_off, a_sz).ljust(64, b"\x00"))

        print(f"[SUCCESS] Compiled to {self.output_path} ({os.path.getsize(self.output_path)} bytes)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--vocab", type=str, help="Path to tokenizer.json")
    args = parser.parse_args()
    
    nc = NodalCompiler("test_model.nbbin")
    if args.vocab:
        nc.load_vocab(args.vocab)
    if args.mock:
        nc.add_tensor("weight_0", np.random.randn(64, 64).astype(np.float32), dtype="NF4")
    nc.compile()
