# INT8 Quantization (Weight-Only Dynamic Dequantization)

This document describes the design, implementation, benchmark findings, and systems trade-offs of our custom INT8 weight-only quantization scheme.

---

## 1. Quantization Concept & Implementation
We implemented **symmetric per-channel (per-row) weight-only INT8 quantization** for LLM linear layers.

```
src/
└── quantization/
    └── int8.py        # INT8Linear layer and recursive model patcher
```

### How it Works
1. **Quantization Formula (Offline Calibration)**:
   For each row $i$ of the weight matrix $W$:
   $$S_i = \frac{\max(|W_i|)}{127}$$
   $$W_{q, i} = \text{round}\left(\frac{W_i}{S_i}\right)$$
   Where $W_{q, i}$ is clamped to $[-128, 127]$ and stored as `torch.int8`. The scale $S_i$ is stored as `torch.float32`.
2. **On-the-fly Dequantization (Forward Pass)**:
   During inference, the input $x$ is in floating-point format. We dequantize the weight tensor back to float right before the linear projection:
   $$W_{\text{dequant}} = W_q \times S$$
   $$Y = X W_{\text{dequant}}^T + b$$

---

## 2. Benchmark Results
We evaluated `Qwen/Qwen2.5-0.5B-Instruct` on CPU using a 128-token context and generating 30 new tokens.

### Performance Summary (CPU Evaluation)

| Metric | FP32 (Baseline) | INT8 (Quantized) | Impact |
|---|---|---|---|
| **Model Load / Quantize Time** | 3.07 s | 0.47 s (quantization overhead) | - |
| **Memory Allocated (Model)** | N/A | N/A | Install psutil to measure CPU RAM |
| **TTFT** | 110.5 ms | 569.6 ms | +415.3% latency |
| **TPOT** | 46.4 ms | 645.5 ms | +1289.8% latency |
| **Decoding Speed** | 21.5 tok/s | 1.5 tok/s | -92.8% speed |
| **Total Latency** | 1457.5 ms | 19289.7 ms | +1223.4% latency |

---

## 3. Analysis & Key Insights

### Why is INT8 Quantization Slower on CPU?
Our benchmark reveals a massive slowdown in decoding speed (from **21.5 tok/s** to **1.5 tok/s**). 
This behavior highlights a critical systems engineering lesson: **without custom hardware-accelerated kernels, quantization increases latency due to dequantization compute overhead.**

1. **Dequantization overhead in PyTorch**: 
   Since we lack custom compiled CPU kernels, PyTorch must allocate a temporary float32 weight matrix and perform element-wise scaling (`weight_q.to(dtype) * scale`) for every forward pass in all 168 linear layers. This adds massive floating-point and memory-management overhead to the CPU at each token step.
2. **Highly Optimized FP32 BLAS**: 
   Standard FP32 linear projections leverage Intel MKL / OpenBLAS libraries which are extremely parallelized and cache-aware on CPU.
3. **GPU / Custom Kernel Parallelism**: 
   On GPU, specialized weight-only kernels (like AWQ or bitsandbytes) perform dequantization on-the-fly *inside* register memory while streaming weights from VRAM to the ALUs. This overlaps memory loading and computation, resulting in a true latency reduction by saving memory bandwidth.

### Memory Savings vs. Throughput Trade-off
* **Memory Savings**: Weight-only INT8 quantization guarantees a theoretical **~50% weight storage reduction** compared to FP16 and **~75% reduction** compared to FP32. For Qwen 2.5-3B, the weights drop from ~6 GB to ~3 GB in RAM/VRAM, enabling local serving on smaller hardware.
* **Throughput**: On standard hardware without customized low-level fused kernels, throughput drops significantly because the GPU/CPU is overwhelmed by the cost of converting weights back to float on every single step.

---

## 4. Interview Questions & Answers

### Q1: What is the difference between asymmetric and symmetric quantization?
* **Symmetric Quantization**: Maps the range of floating-point values $[-max(|x|), max(|x|)]$ symmetrically to integer values $[-127, 127]$ (or $[-128, 127]$). The zero-point is fixed at 0. The scale factor is $S = \max(|x|) / 127$. It is computationally cheaper since there is no zero-point offset to add or subtract during matrix multiplication.
* **Asymmetric Quantization**: Maps the actual min/max range $[min(x), max(x)]$ to the full integer range $[0, 255]$ (or $[-128, 127]$). It uses both a scale factor $S = (max - min) / 255$ and a non-zero integer **zero-point** ($Z = \text{round}(-min / S)$) representing float 0.0. It is more accurate for skewed distributions but introduces calculation overhead in computing $(W_q - Z_w)(X_q - Z_x)$.

### Q2: Why is the final projection layer (`lm_head`) typically excluded from LLM quantization?
The output vocabulary projection layer (`lm_head`) maps the hidden states (e.g., dimension 2048) to the vocabulary dimension (e.g., 151,936 for Qwen). 
1. **High Precision Requirements**: The final logit distribution is highly sensitive; quantizing this layer leads to a massive increase in perplexity and causes the model to generate repetitive or incoherent tokens.
2. **Low VRAM footprint contribution**: Since the layer is only run once at the end of the transformer stack and does not scale with model depth, keeping it in high precision (FP16/FP32) preserves model accuracy at negligible VRAM cost.

### Q3: Why does Weight-Only Quantization dequantize weights back to Float during forward pass instead of quantizing the activations to Integer?
LLM activations contain severe outlier channels (values that are 100x larger than average, especially in models $>6.7\text{B}$ parameters). Quantizing activations to INT8 forces the scale factor to be large, which destroys the precision of all other activation channels and ruins accuracy.
Weight-only quantization bypasses this issue entirely because LLM weights do not have extreme outliers. Keeping activations in high precision (FP16/FP32) and dequantizing weights dynamically preserves model accuracy. Weight-to-Activation Quantization (W8A8) requires complex outlier management schemes like SmoothQuant to prevent quality degradation.
