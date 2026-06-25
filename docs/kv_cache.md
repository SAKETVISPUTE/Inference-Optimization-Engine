# Key-Value (KV) Cache Optimization

This document provides a detailed breakdown of the Key-Value (KV) Cache implementation, complexity analysis, benchmark findings, and systems-level interview preparation.

---

## 1. Intuition: Why Autoregressive Generation is Slow
In causal language modeling, to predict the next token $x_{t}$, the attention mechanism computes relationships between the Query vector of the current token and the Key/Value vectors of all previous tokens in the sequence ($x_1, x_2, \dots, x_{t-1}$).
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

At step $t$, the representations of past tokens ($x_1, \dots, x_{t-1}$) do not change. 
* **Without KV Cache**: We recompute the Key ($K$) and Value ($V$) states for every past token from scratch at every step. This leads to redundant matrix-matrix multiplications at every layer.
* **With KV Cache**: We compute the $K$ and $V$ tensors for the past tokens only once. We store them in VRAM. For subsequent tokens, we only compute $Q, K, V$ for the single *new* token, append the new $K$ and $V$ to our cache, and perform attention against the full cache.

---

## 2. Implementation details

```
src/
└── cache/
    └── cache_manager.py  # Tracks cache states and calculates memory footprint
```

### KV Cache Decode Step Flow
1. **Prefill**: Pass the full prompt of size $P$. Save the resulting $K$ and $V$ states for all $P$ tokens in the cache.
2. **Decode Step**: 
   * Forward pass inputs are reduced to only the single last generated token (shape: `[batch_size, 1]`).
   * Pass the cached `past_key_values` into the model's forward call:
     ```python
     outputs = model(input_ids=next_input_ids, past_key_values=past_key_values, use_cache=True)
     ```
   * Retrieve the new logits and the updated cache containing the new token's $K$ and $V$ states appended to the old ones.

---

## 3. Computational Complexity Analysis

Let $P$ be the prompt length, and $M$ be the number of new tokens generated.

| Metric | Without KV Cache | With KV Cache |
|---|---|---|
| **Attention Math Flops** | $O((P+M)^3)$ (Quadratic per step) | $O(M \cdot (P+M))$ (Linear per step) |
| **Model Weights Loaded** | $O(M \cdot (P+M))$ (Loads entire model weights multiple times per token) | $O(M)$ (Loads entire model weights exactly once per token) |
| **VRAM Footprint (Cache)** | $O(1)$ (No memory overhead) | $O(B \cdot L \cdot H \cdot D \cdot N_{seq})$ (Grows linearly) |

### Memory Footprint Equation
The memory footprint (in bytes) of the KV Cache is:
$$\text{KV Cache Size} = 2 \times \text{Num Layers} \times \text{Num KV Heads} \times \text{Head Dimension} \times \text{Sequence Length} \times \text{Batch Size} \times \text{Bytes Per Element}$$

For a model with $L$ layers, $H_{kv}$ key-value heads, $D$ head dimension, batch size $B$, and sequence length $S$, we store both Keys and Values (hence the factor of 2). Under FP16/BF16, `Bytes Per Element = 2`.

---

## 4. Benchmark Results
We benchmarked `Qwen/Qwen2.5-0.5B-Instruct` on CPU. The results quantitatively demonstrate how the generation speed (TPOT) degrades under sequence length growth without caching, while remaining constant with KV Cache enabled.

### CPU Benchmark Performance Summary

| Prompt Length | Cache Mode | TTFT (ms) | TPOT (ms) | Speed (tok/s) | Peak Mem Delta | Speedup |
|---|---|---|---|---|---|---|
| **Short (16 tokens)** | No Cache | 102.7 | 198.1 | 5.0 | 0.00 MB | baseline |
| **Short (16 tokens)** | KV Cache | 123.5 | 82.3 | 12.1 | 0.00 MB | **2.41x** |
| | | | | | | |
| **Medium (128 tokens)**| No Cache | 421.8 | 358.6 | 2.8 | 0.00 MB | baseline |
| **Medium (128 tokens)**| KV Cache | 211.5 | 60.7 | 16.5 | 0.00 MB | **5.90x** |
| | | | | | | |
| **Long (256 tokens)** | No Cache | 296.7 | 427.7 | 2.3 | 0.00 MB | baseline |
| **Long (256 tokens)** | KV Cache | 385.4 | 60.7 | 16.5 | 0.00 MB | **7.05x** |

### Key Observations
1. **Constant Decoding Speed**: With KV Cache enabled, the decoding speed (TPOT) remains flat at **~60.7 ms** (16.5 tok/s) between 128 and 256 context lengths.
2. **Degrading Baseline Speed**: Without KV Cache, decoding slows from **198.1 ms** (5.0 tok/s) to **427.7 ms** (2.3 tok/s), showing the quadratic slowdown.
3. **Speedup Growth**: The KV cache speedup grows from **2.41x** to **7.05x** as context length expands.

---

## 5. Common Bugs & Systems Pitfalls

1. **Incorrect Attention Mask / Position IDs**: If `position_ids` are not updated to match the total cached sequence length, the model applies rotary embeddings (RoPE) using incorrect index coordinates, resulting in garbled text.
2. **Cache Shape Mismatch**: Concatenating keys and values along the batch or head dimension instead of the sequence length dimension.
3. **Memory Fragmentation (OOM)**: Since KV caches grow dynamically, VRAM gets fragmented over time, leading to Out-Of-Memory (OOM) errors even when total free VRAM appears sufficient.

---

## 6. Interview Questions & Answers

### Q1: Calculate the KV cache memory footprint for Llama 3-8B at batch size 16 and sequence length 8192 in FP16.
Llama 3-8B architecture details:
* Layers ($L$) = 32
* Key-Value Heads ($H_{kv}$) = 8 (Grouped Query Attention)
* Head Dimension ($D$) = 128
* Element size = 2 bytes (FP16)

Apply the formula:
$$\text{Memory} = 2 \times 32 \text{ layers} \times 8 \text{ KV heads} \times 128 \text{ dim} \times 8192 \text{ seq} \times 16 \text{ batch} \times 2 \text{ bytes}$$
$$\text{Memory} = 2 \times 32 \times 8 \times 128 \times 8192 \times 16 \times 2 = 17,179,869,184 \text{ bytes} \approx 17.18 \text{ GB}$$

> [!IMPORTANT]
> The KV Cache alone requires ~17.18 GB of VRAM. This is larger than the model weights themselves (~16 GB for 8B in FP16), illustrating why KV Cache capacity is the primary serving bottleneck.

### Q2: What is Grouped Query Attention (GQA), and how does it optimize KV Cache overhead?
In Multi-Head Attention (MHA), every query head has its own key and value head. This leads to high KV Cache memory consumption.
In Multi-Query Attention (MQA), all query heads share a single key and value head. This dramatically reduces cache size (by $H$ times) but degrades model capacity and accuracy.
Grouped Query Attention (GQA) is a middle-ground optimization. Query heads are divided into groups, and each group shares a single KV head. For example, Llama 3 has 32 query heads and 8 KV heads (group ratio of 4:1). GQA achieves near-MHA accuracy while reducing the KV Cache memory footprint and memory-bandwidth overhead by 4x.

### Q3: What is KV Cache fragmentation, and how does PagedAttention solve it?
In standard serving, VRAM for a request's KV Cache must be allocated contiguously to match the maximum sequence length. Because sequence lengths are dynamic, this causes:
1. **Internal Fragmentation**: Allocating VRAM up to `max_seq_len` for a request that ends early.
2. **External Fragmentation**: Virtual memory slots are interleaved, preventing allocation.
PagedAttention (used in vLLM) solves this by partitioning the KV cache into small, fixed-size physical blocks (similar to paging in OS kernel memory management). The block tables map logical sequences to non-contiguous physical blocks, achieving near 0% VRAM waste and enabling higher batch sizes.
