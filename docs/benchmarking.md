# GPU Benchmarking Report

This document presents the performance benchmarking results of our custom LLM inference engine using **Qwen 2.5-3B Instruct** on an NVIDIA RTX A6000 GPU (48GB VRAM).

---

## 1. Benchmark Methodology

We compared four system configurations:
1. **FP32 (No Cache)**: Causal language model running FP32 without Key-Value caching.
2. **FP32 (KV Cache)**: Causal language model running FP32 with our custom iteration-level KV Cache wrapper.
3. **INT8 (KV Cache)**: Weight-only linear layer quantization (quantized down to 8-bit, dynamically dequantized to FP32 during forward pass) with KV Cache.
4. **INT4 (KV Cache)**: Weight-only 4-bit packed linear layer quantization (dynamically unpacked during forward pass) with KV Cache.

### Swept Parameters:
- **Context Length (Prompt size)**: `[512, 2048, 4096, 8192, 16384]`
- **Output Length (New tokens)**: `[32, 128, 512, 1024]`

---

## 2. Core Results (Context = 2048, Output = 128)

| Configuration | TTFT (ms) | TPOT (ms) | Speed (tok/s) | Peak VRAM Delta (GB) | Total Latency (ms) |
|---|---|---|---|---|---|
| **FP32 (No Cache)** | 458.6 | 502.6 | 1.99 | 2.78 | 64286.5 |
| **FP32 (KV Cache)** | 457.2 | 28.4 | 35.25 | 1.35 | 4060.0 |
| **INT8 (KV Cache)** | 566.0 | 45.5 | 21.96 | 1.47 | 6349.0 |
| **INT4 (KV Cache)** | 731.8 | 106.2 | 9.41 | 1.51 | 14225.5 |

---

## 3. Key Findings & Systems Analysis

### 1. The Impact of KV Caching
The transition from **No Cache** to **KV Cache** under FP32 at a context size of 2048 yields a massive **17.7x speedup** (climbing from `1.99` tok/s to `35.25` tok/s). 
- Without caching, attention complexity scales quadratically ($O(N^2)$), requiring re-computation of key/value activations for all past tokens at every decoding step.
- Caching bounds sequence decoding complexity to $O(N)$ for attention math, maintaining a flat Time Per Output Token (TPOT) of ~28ms regardless of generation length.

### 2. Quantization Tradeoffs in Pure PyTorch
Quantization decreases memory consumption but **slows down generation** in a standard PyTorch runtime environment:
- **INT8 (KV Cache)** runs at `21.96` tok/s (a 37.7% slowdown vs. FP32).
- **INT4 (KV Cache)** runs at `9.41` tok/s (a 73.3% slowdown vs. FP32).
- *Reason*: Standard PyTorch executes dynamic dequantization (casting and scaling integer weights back to float) inside standard Python/PyTorch operations. The overhead of allocation and element-wise math offsets the reduction in weight loading size. In production (e.g. TensorRT-LLM, AWQ), low-level CUDA kernels perform fused matrix multiplication directly on quantized integer data inside GPU registers, bypassing memory allocations.

### 3. VRAM Memory Bounds & OOM Prevention
At a massive context size of **16,384 tokens**, the FP32 model experiences a **CUDA Out of Memory (OOM)** crash. However, the quantized **INT8** and **INT4** models run successfully, restricting peak VRAM requirements to **39.0 GB** and avoiding OOM. This demonstrates the critical memory-saving utility of quantization for long-context workloads.
