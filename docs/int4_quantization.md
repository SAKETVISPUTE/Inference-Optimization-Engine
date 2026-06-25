# INT4 Quantization (Weight-Only Packed Dequantization)

This document describes the implementation, benchmarking, and systems analysis of our custom INT4 weight-only packed quantization scheme.

---

## 1. Quantization Concept & Implementation
In memory-bandwidth bound LLM serving, compressing weights to 4-bits provides an additional 2x memory savings over INT8 and an 8x reduction over FP32. However, since computer hardware works with 8-bit bytes, storing 4-bit numbers individually does not save storage memory. We must pack them.

```
src/
└── quantization/
    └── int4.py        # INT4Linear layer with bit-packing and unpacking logic
```

### Weight-Packing Algorithm (Offline)
For a weight matrix of shape $[Out, In]$, we quantize rows symmetrically into 4-bit values.
1. **Symmetric Row-wise Scaling**:
   $$S_i = \frac{\max(|W_i|)}{7.0}$$
   Map weights to signed integer range $[-8, 7]$:
   $$W_{q, i} = \text{round}\left(\frac{W_i}{S_i}\right)$$
2. **Shift to Unsigned Range**: 
   $$W_{u} = W_{q} + 8 \quad (\text{range } [0, 15])$$
3. **Column-wise Packing**: We pack two adjacent columns into a single 8-bit byte (`uint8`).
   * High 4 bits (nibble): Even columns (`weight_u[:, 0::2]`)
   * Low 4 bits (nibble): Odd columns (`weight_u[:, 1::2]`)
   $$W_{\text{packed}} = (W_{\text{even}} \ll 4) \mid W_{\text{odd}}$$
   The resulting tensor has shape $[Out, In / 2]$ and data type `torch.uint8`, saving exactly 50% storage space compared to INT8.

### Unpacking & Dequantization (Forward Pass)
During inference, right before execution:
1. **Bitwise Unpacking**:
   $$W_{\text{even}} = (W_{\text{packed}} \gg 4) \& 0\text{x0F}$$
   $$W_{\text{odd}} = W_{\text{packed}} \& 0\text{x0F}$$
2. **Reconstruction**: Allocate a tensor of original shape $[Out, In]$ and interleave the columns:
   $$W_u[:, 0::2] = W_{\text{even}}$$
   $$W_u[:, 1::2] = W_{\text{odd}}$$
3. **Signed Scale Reconstruction**:
   $$W_{\text{dequant}} = (W_u - 8) \times S$$

---

## 2. Comparative Benchmark Results
We benchmarked FP32, INT8, and INT4 configurations on CPU using `Qwen/Qwen2.5-0.5B-Instruct` (128 context tokens, 30 generated tokens, with KV Cache enabled).

### CPU Evaluation Metrics

| Configuration | Memory Footprint (Model) | TTFT (ms) | TPOT (ms) | Speed (tok/s) | Latency (ms) |
|---|---|---|---|---|---|
| **FP32 (Baseline)** | N/A | 164.3 | 49.6 | 20.2 | 1603.2 |
| **INT8 (Quantized)** | N/A | 385.1 | 362.9 | 2.8 | 10908.2 |
| **INT4 (Quantized)** | N/A | 773.6 | 723.0 | 1.4 | 21741.3 |

---

## 3. Systems-Level Performance Analysis

### Why is INT4 Slower than INT8 and FP32 on CPU?
The decoding speed degrades from **20.2 tok/s** (FP32) to **2.8 tok/s** (INT8) and down to **1.4 tok/s** (INT4).
* **Increased CPU Instruction Overhead**: 
  To run an INT4 layer, the CPU must compute bitwise right-shifts (`>>`), bitwise ANDs (`&`), allocates a temporary matrix of the full size, and copy slices of even/odd columns. These Python/PyTorch loops run for all 168 layers at every token generation step.
* **Lack of Fused Kernels**: 
  In high-performance libraries like `vLLM` or `TensorRT-LLM` on GPU, bitwise unpacking and dequantization are implemented in low-level fused CUDA/C++ kernels. The weights are unpacked inside registers or L1 shared memory right before vector multiplication, avoiding virtual memory allocation overhead entirely. 
* **The Compression vs Latency Trade-off**: 
  Quantization without fused hardware kernels is purely a **space optimization** (reduces weights VRAM/RAM footprint by 8x) but acts as a **latency bottleneck** due to instruction and allocation overhead.

### Model Coherence Degradation (0.5B Scale)
While the INT4 model correctly generates the first sentence (*"The capital of France is Paris."*), it later degrades into repetitive formatting patterns like `( ) ( )`. 
* **Reasoning**: Standard row-wise quantization maps the entire row (size 2048/4096) to a single scale factor. For a tiny 0.5B model, the representation capacity of 4-bit (only 15 integer values) is too coarse, resulting in quantization noise. 
* **Production Fix**: Production 4-bit serving (like AWQ or GPTQ) uses **grouped quantization** (group size 128 or 64) where a separate scale factor is calculated for every block of 128 weights, preserving accuracy.

---

## 4. Interview Questions & Answers

### Q1: What is weight packing, and why is it necessary for INT4/INT2 quantization?
Standard memory architectures are byte-addressable. The smallest addressable unit of storage is 8 bits (1 byte). If we store 4-bit values inside standard 8-bit registers or tensors, the remaining 4 bits are padded with zeros, which wastes half the space and defeats the purpose of compression. 
Weight packing consolidates multiple lower-bit values into a single 8-bit byte (e.g., two 4-bit weights packed into one `uint8`, or four 2-bit weights packed into one `uint8`). This reduces storage and memory bandwidth transfer sizes by 2x or 4x.

### Q2: How does Grouped Quantization (e.g. Group Size 128) preserve LLM accuracy in INT4?
In per-channel (per-row) quantization, we calculate one scaling factor per row. If a row contains a few large weights (outliers), the scaling factor becomes large. This squashes all the other smaller weights in that row down to 0, destroying precision.
Grouped quantization divides each row of size $N$ into small blocks of size $G$ (typically $G=128$ or $G=64$). A separate scaling factor is calculated for each group of 128 weights. This confines outliers to their local group, keeping the scaling factor small for all other groups and preserving the dynamic range and perplexity of the model.

### Q3: Why does memory bandwidth limit autoregressive decoding speed, and how does INT4 help on GPUs?
During autoregressive decoding, batch size is small (often 1). The arithmetic intensity (ratio of FLOPs to memory bytes read) is extremely low because we perform vector-matrix multiplications (GEMV). The GPU ALUs are idle waiting for the model weights to load from VRAM to SRAM.
INT4 quantization reduces the size of the weights by 4x compared to FP16. This means we only need to transfer 1/4th the bytes from VRAM to the GPU processor. On GPUs with optimized dequantization kernels, unpacking the bits in cache takes negligible cycles compared to VRAM read latencies, which yields a near 4x throughput speedup under memory-bandwidth bound scenarios.
