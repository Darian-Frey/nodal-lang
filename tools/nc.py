import struct
import numpy as np
import argparse
import os

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

    def add_tensor(self, name, data, dtype="NF4", block_size=64):
        if dtype == "NF4":
            packed, scales = quantize_nf4(data, block_size)
            # NF4 Aux Header: has_z(1), scale_dtype(1), z_dtype(1), res(1), blk_sz(4), num_blk(4)
            aux_header = struct.pack("<BBBBII", 0, 0, 0, 0, block_size, len(scales))
            self.tensors.append({
                "name": name, "data": packed, "aux_data": aux_header + scales.tobytes(),
                "dtype": 4, "shape": list(data.shape), "block_size": block_size
            })
        else:
            self.tensors.append({
                "name": name, "data": data.tobytes(), "aux_data": None,
                "dtype": 0, "shape": list(data.shape)
            })

    def compile(self):
        with open(self.output_path, "wb") as f:
            # 1. Header (32 bytes)
            f.write(b"\x00" * 32) 
            tensor_table_offset = f.tell()
            
            # 2. Tensor Table Placeholder
            for _ in self.tensors: f.write(b"\x00" * 64)

            # 3. Data & Aux Segments (Aligned to 64 bytes)
            locs = []
            for t in self.tensors:
                # Data Segment
                f.write(b"\x00" * (align_to(f.tell()) - f.tell()))
                d_off = f.tell()
                f.write(t["data"])
                d_sz = len(t["data"])
                
                # Aux Segment
                a_off, a_sz = 0, 0
                if t["aux_data"]:
                    f.write(b"\x00" * (align_to(f.tell()) - f.tell()))
                    a_off = f.tell()
                    f.write(t["aux_data"])
                    a_sz = len(t["aux_data"])
                
                locs.append((d_off, d_sz, a_off, a_sz))

            # 4. Final Patching
            f.seek(0)
            f.write(struct.pack("<IHHIIQ", 0x4E42444E, 1, 0, len(self.tensors), tensor_table_offset, 0))
            f.seek(tensor_table_offset)
            for i, t in enumerate(self.tensors):
                d_off, d_sz, a_off, a_sz = locs[i]
                # NDBN_TensorEntry (64 bytes)
                f.write(struct.pack("<IBBBB4IQQQQ", 
                    0, t["dtype"], len(t["shape"]), 0, 1 if t["aux_data"] else 0,
                    *(t["shape"] + [0]*(4-len(t["shape"]))), 
                    d_off, d_sz, a_off, a_sz).ljust(64, b"\x00"))

        print(f"[SUCCESS] {self.output_path} generated ({os.path.getsize(self.output_path)} bytes)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    if args.mock:
        nc = NodalCompiler("test_model.nbbin")
        nc.add_tensor("weight_0", np.random.randn(64, 64).astype(np.float32), dtype="NF4")
        nc.compile()
