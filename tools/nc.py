"""
nc.py - The Nodal Compiler (Prototype)
Converts models to .nbbin and generates Nodal IR tapes.
"""

import struct
import numpy as np
import argparse
import os

def align_to(size, alignment=64):
    return (size + alignment - 1) & ~(alignment - 1)

class NodalCompiler:
    def __init__(self, output_path):
        self.output_path = output_path
        self.tensors = []
        self.string_table = b""

    def add_tensor(self, name, data, dtype="NF4"):
        # Placeholder for actual NF4 quantization logic
        # For alpha, we assume 'data' is already a byte array or numpy array
        self.tensors.append({
            "name": name,
            "data": data.tobytes() if hasattr(data, 'tobytes') else data,
            "dtype": 4 if dtype == "NF4" else 0,
            "shape": list(data.shape) if hasattr(data, 'shape') else [len(data), 1, 0, 0]
        })

    def compile(self):
        with open(self.output_path, "wb") as f:
            # 1. Header Placeholder (32 bytes)
            f.write(struct.pack("<IHHIIQ", 0, 0, 0, 0, 0, 0))
            f.write(b"\x00" * 8) # Padding

            tensor_table_offset = f.tell()
            
            # 2. Tensor Table Placeholder
            # Each entry is 64 bytes
            for _ in self.tensors:
                f.write(b"\x00" * 64)

            # 3. Data Segments (Aligned to 64 bytes)
            entries = []
            for t in self.tensors:
                # Align the file pointer
                padding = align_to(f.tell()) - f.tell()
                f.write(b"\x00" * padding)
                
                offset = f.tell()
                f.write(t["data"])
                size = len(t["data"])
                
                entries.append((offset, size))

            # 4. Rewrite Header & Table with actual offsets
            f.seek(0)
            f.write(struct.pack("<IHHIIQ", 
                0x4E42444E, # Magic 'NDBN'
                1,          # Version
                0,          # Flags
                len(self.tensors),
                tensor_table_offset,
                0           # String Table Offset
            ))

            f.seek(tensor_table_offset)
            for i, t in enumerate(self.tensors):
                offset, size = entries[i]
                # Entry: NameOff(4), DType(1), Rank(1), Layout(1), Q(1), Shape(16), DataOff(8), DataSize(8), ...
                entry = struct.pack("<IBBBB4IQQ", 
                    0, t["dtype"], len(t["shape"]), 0, 0,
                    *t["shape"], offset, size
                )
                f.write(entry.ljust(64, b"\x00"))

        print(f"[SUCCESS] Compiled {len(self.tensors)} tensors to {self.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nodal Compiler")
    parser.add_argument("--mock", action="store_true", help="Generate a mock .nbbin for testing")
    args = parser.parse_args()

    if args.mock:
        nc = NodalCompiler("test_model.nbbin")
        # Create a mock 4x4 tensor
        mock_data = np.random.rand(4, 4).astype(np.float32)
        nc.add_tensor("mock_weight", mock_data, dtype="F32")
        nc.compile()
