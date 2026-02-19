# Contributing to Nodal

First off, thank you for your interest in Nodal! By contributing to this project, you are helping build the foundation for high-performance, edge-native AI.

## 🛠️ Where We Need Help

Nodal is in an early alpha stage. We are currently prioritizing the following areas:

### 1. Hardware Kernels (`src/kernels/`)
We want Nodal to run on everything. We need optimized C/Assembly kernels for:
* **RISC-V Vector (RVV)**
* **AVX-512 (x86)**
* **Apple Silicon (Accelerate/AMX)**
* **CUDA/Tensor Cores** (Host-side orchestration)

### 2. Compiler Expansion (`tools/nc.py`)
* Support for more model formats (GGUF, Safetensors, ONNX).
* Advanced quantization methods (IQ4_XS, GGUF-style k-quants).

### 3. Documentation & Examples
* Tutorials for running Nodal on specific hardware (Raspberry Pi, ESP32-S3, etc.).
* Sample "Nodalized" models for the community to test.

## 📏 Technical Standards

To maintain our "Bare-Metal" promise, all code contributions must follow these rules:

* **Zero Dynamic Allocation:** No `malloc`, `free`, or `new` in the inference hot-path. Use the provided Arena buffers.
* **Deterministic Logic:** Avoid non-deterministic behavior. Ensure bit-perfect results across different CPU architectures where possible.
* **Minimal Dependencies:** Kernels should only depend on `nodal.h` and standard math libraries.
* **Alignment Matters:** Ensure all data accesses respect the 64-byte alignment contract of the `.nbbin` format.

## 🚀 How to Submit a Change

1. **Fork the Repo:** Create your own copy of `nodal-lang`.
2. **Create a Branch:** `git checkout -b feat/your-feature-name`.
3. **Write Tests:** If you add a kernel, include a small test in `examples/` to verify its math.
4. **Submit a PR:** Provide a clear description of what changed and why. Reference any related issues.

## 💬 Communication
If you have questions about the architecture or want to propose a major change, please open a **GitHub Issue** with the prefix `[PROPOSAL]`.

---
*Welcome to the metal. Let's build the future of edge AI together.*
