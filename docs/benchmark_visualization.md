# 📊 Benchmark Visualizations

I have parsed the GPU benchmark results (`gpu_benchmark_results.json`) for a context length of **512 tokens** and generation length of **128 tokens** and plotted the metrics below.

---

## 📈 Throughput & Memory Comparison

![Benchmark Comparison Plots](benchmark_plots.png)


---

## 🔍 Key Insights from the Charts

1. **KV Cache Optimization (FP32 (KV Cache) vs FP32 (No Cache))**:
   * **Throughput**: Enabling the KV cache increases generation speed from **~8.4 tokens/sec** to **~34.4 tokens/sec** (more than a **4x speedup**!).
   * **Reason**: Without caching, the attention matrix recalculates the entire prompt and all generated tokens at every step. With caching, it only calculates a single token and loads the rest from memory.

2. **Quantization Memory Savings (INT8 and INT4 vs FP32)**:
   * **VRAM footprint**: 
     * **FP32** requires **0.34 GB** (excluding base model weights).
     * **INT8** drops this VRAM requirement significantly.
     * **INT4** provides the absolute minimum memory footprint, showing the power of 4-bit packing.
   * **The Latency Tradeoff**: In pure PyTorch (without custom compiled CUDA kernels), you will notice that INT8 and INT4 have lower tokens/sec than FP32. This is due to the **dynamic dequantization overhead** (PyTorch casting and scaling tensors in Python at every forward layer pass).
