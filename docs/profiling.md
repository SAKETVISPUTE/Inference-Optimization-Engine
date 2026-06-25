# GPU Profiling Report

This document outlines the findings from our profiling experiments using `torch.profiler` and Chrome Tracing logs (Perfetto).

---

## 1. Profiling Methodology

We profiled a short generation run of 16 output tokens across our four baseline configurations:
- `fp32_nocache_trace.json`
- `fp32_cache_trace.json`
- `int8_cache_trace.json`
- `int4_cache_trace.json`

Traces were saved in Chrome Tracing JSON format and reviewed in Perfetto to analyze CPU host execution, kernel launch bottlenecks, and VRAM memory footprint signatures.

---

## 2. Trace Signatures & Analysis

### 1. FP32 No-Cache Trace (`fp32_nocache`)
- **Visual Pattern**: An escalating step staircase. At each iteration, the execution blocks for attention calculation become progressively longer.
- **CPU Side**: Host spends significant time in `aten::add` and linear layer operations, recalculating keys and values for all past context tokens.
- **GPU Side**: Attention kernel execution times grow quadratically, showing massive `cudaLaunchKernel` blocks.

### 2. FP32 With-Cache Trace (`fp32_cache`)
- **Visual Pattern**: A single long prefill block, followed by 16 identical, rapid, highly uniform decode blocks.
- **CPU Side**: High CPU activity in the prefill stage, transitioning to minimal overhead during decode.
- **GPU Side**: Single-token attention kernels are extremely fast, but show launch gaps. The time between GPU kernel executions (host-device synchronization) represents a minor bottleneck.

### 3. INT8 Cache Trace (`int8_cache`)
- **Visual Pattern**: Decode blocks are significantly wider than the FP32 cache trace, with high density of CPU-bound operations.
- **Trace Details**: We observe massive occurrences of `aten::copy_`, `aten::to`, and element-wise `aten::mul` inside the linear layers. This represents the dynamic dequantization cost (loading quantized int8 weights, allocating memory, and scaling them back to float).
- **GPU Bottleneck**: The Tensor Cores sit idle waiting for these dequantization operations to complete on the CPU host.

### 4. INT4 Cache Trace (`int4_cache`)
- **Visual Pattern**: Heavily fragmented trace blocks with high latency.
- **Trace Details**: Trace shows frequent bit-shift and masking operators (`aten::bitwise_and`, `aten::right_shift`) at every single linear layer layer. 
- **Analysis**: Unpacking 4-bit packed weights dynamically in PyTorch requires many tiny element-wise operations. These operations are launched sequentially from Python, creating a severe CPU-side bottleneck and kernel queue delay on the GPU.

---

## 3. Systems Summary

The traces confirm that:
1. **KV Caching** is essential to prevent CPU/GPU work scaling with context length.
2. **Quantization** (INT8/INT4) requires **fused kernels** (where unpacking and math occur in a single GPU instruction block) to achieve speedups. Dynamic dequantization inside standard PyTorch operators degrades execution efficiency.
