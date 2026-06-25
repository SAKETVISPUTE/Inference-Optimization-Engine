# Continuous Batching Scheduler

This document explains the problem, design, implementation, and performance characteristics of our Continuous Batching Scheduler.

---

## 1. The Problem: Static vs. Continuous Batching

In LLM serving, autoregressive generation has highly variable sequence lengths. 
- **Sequential Serving**: Low throughput because weight memory bandwidth dominates.
- **Static Batching**: Grouping requests and padding them to the same length. This causes two major performance bottlenecks:
  1. **Padding Overhead**: Computation is wasted on padding tokens.
  2. **Long-Tail Latency**: If one request in the batch needs 500 tokens and another needs only 5 tokens, the first request is held hostage for the entire duration of the longer request, causing high latency for the user.

**Continuous Batching** (iteration-level scheduling) processes requests at the *token iteration level*. Active requests are evicted as soon as they generate an `EOS` token, and pending requests are immediately scheduled into the free batch slots.

---

## 2. Architecture & Design

Our implementation resides in `src/scheduler/` and consists of the following components:

```
Request Queue ──► Scheduler (Iteration-Level Control) ──► Batch Manager
                      ▲                                        │
                      └───────────────── Step output ──────────┘
```

- **[request.py](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/src/scheduler/request.py)**: Represents a single client request, holding its ID, prompt tokens, generated tokens, status (`PENDING`, `RUNNING`, `FINISHED`), and timing metrics (arrival, start, and finish times).
- **[request_queue.py](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/src/scheduler/request_queue.py)**: Manages incoming requests, maintaining the FIFO queue of pending work.
- **[batch_manager.py](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/src/scheduler/batch_manager.py)**: Manages the active slot allocation in the execution batch up to `max_batch_size`.
- **[scheduler.py](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/src/scheduler/scheduler.py)**: Implements the serving execution loops:
  - `run_sequential`: One-by-one execution.
  - `run_static_batch`: Left-padding static batch execution.
  - `run_continuous_batch`: Token-level continuous scheduling.

---

## 3. Benchmark Results & Analysis

Below are the quantitative results comparing Sequential, Static, and Continuous batching configurations on an NVIDIA RTX A6000 GPU (48GB VRAM) using **Qwen 2.5-3B Instruct**:

### Performance Comparison Table

| Mode | Concurrency | Throughput (tok/s) | Avg Latency (ms) | P95 Latency (ms) | VRAM (GB) |
|---|---|---|---|---|---|
| **Sequential** | 1 | 29.59 | 2162.9 | 2162.9 | 6.55 |
| | 2 | 33.44 | 3116.7 | 4187.1 | 6.55 |
| | 4 | 32.49 | 6210.0 | 10334.2 | 6.55 |
| | 8 | 33.23 | 13671.4 | 27003.4 | 6.55 |
| **Static Batch** | 1 | 32.64 | 2020.7 | 2020.7 | 6.55 |
| | 2 | 57.48 | 2259.5 | 2481.0 | 6.62 |
| | 4 | 99.53 | 2785.5 | 3462.5 | 6.74 |
| | 8 | 163.42 | 4081.7 | 5697.4 | 7.01 |
| | 16 | 296.00 | 4464.5 | 6494.2 | 7.54 |
| | 32 | 520.13 | 4895.3 | 7395.3 | 8.60 |
| **Continuous** | 1 | 33.45 | 1933.7 | 1933.7 | 6.55 |
| | 2 | 31.29 | 4505.4 | 4734.9 | 6.62 |
| | 4 | 32.37 | 9628.5 | 10797.1 | 6.74 |
| | 8 | 32.70 | 24309.8 | 29299.6 | 7.01 |
| | 16 | 32.94 | 39354.0 | 57719.3 | 7.01 |
| | 32 | 33.65 | 67218.8 | 111593.0 | 7.01 |

### Key Observations & Systems Analysis

1. **Static Batching Throughput Scaling**:
   As concurrency scales from 1 to 32, static batching throughput grows almost **16x** (from `32.64` to `520.13` tok/s). This demonstrates the transition from a memory-bandwidth-bound decode phase to a compute-saturated utilization state. The GPU loads the weights once and uses them across 32 sequences, amortizing the weight loading latency.
   
2. **Sequential Queue Congestion**:
   Under sequential serving, throughput remains flat at ~33 tok/s, while P95 latency degrades exponentially (from `2.1s` to `27s` at concurrency 8) because requests spend significant time waiting in the FIFO queue.
   
3. **Continuous Batching Logical Scheduling vs. GPU Fused Kernels**:
   - In this simplified implementation, the active continuous batch requests are executed in a sequential Python loop at each token generation step. As a result, the measured execution time is the sum of the individual step times (resulting in throughput matching sequential).
   - In production engines (e.g., vLLM), the scheduler combines active decode sequences into a single fused batch execution step via **PagedAttention**, achieving the **high throughput of static batching** (520+ tok/s) while preserving the low latency and queue dynamics of continuous scheduling.
   - Continuous batching prevents VRAM memory exhaustion by enforcing a strict slot limit (`max_batch_size=8`), queueing excessive requests until slots are freed by completed requests.


---

## 4. Interview Prep: 20 System Design & Serving Questions

### Q1: What is the primary difference between static batching and continuous batching in LLM serving?
**Answer**: Static batching groups requests at the sequence level; the batch runs until the longest request completes, wasting compute on padding. Continuous batching schedules at the iteration (token) level, evicting completed requests and inserting new ones at each forward pass.

### Q2: Why is LLM generation memory-bandwidth bound rather than compute-bound during the decoding phase?
**Answer**: During decoding, we generate one token at a time. The model needs to load all its weights (e.g., 6GB for a 3B model) from High Bandwidth Memory (HBM) to SRAM to process a single token input. The arithmetic intensity (math operations per byte of memory read) is extremely low, meaning execution speed is limited by how fast weights can be read from memory.

### Q3: How does continuous batching help saturate GPU compute utilization?
**Answer**: By batching multiple requests, we load the model weights once and reuse them for multiple inputs (batch size $B$). This scales the arithmetic intensity by $B$, moving the workload from being memory-bound towards being compute-bound and utilizing the GPU Tensor Cores more effectively.

### Q4: Explain the prefill vs. decoding phase latency characteristics.
**Answer**: 
- **Prefill**: Processes the entire input prompt in parallel. It is compute-bound because we perform matrix multiplications of size `(seq_len, hidden_dim)`.
- **Decoding**: Processes one token at a time sequentially. It is memory-bound due to weight-loading latency.

### Q5: What is "In-Flight Batching" (or chunked prefill)?
**Answer**: In continuous batching, a new request's prefill phase must run alongside the decoding phases of existing requests. In-flight batching merges the long prefill prompt and the single decode tokens into a single execution step to avoid blocking decodes.

### Q6: What role does the KV Cache play in continuous batching scheduling?
**Answer**: The KV Cache stores past Key and Value activations so we don't recompute them. The scheduler must allocate and track memory blocks for each active request's KV Cache. Because lengths vary, KV cache memory footprint is highly dynamic and can cause OOM errors if unmanaged.

### Q7: Explain why PagedAttention is needed in production continuous batching systems like vLLM.
**Answer**: Standard KV caches require contiguous virtual memory allocations. This leads to severe fragmentation and overallocation (since we must reserve memory up to `max_seq_len`). PagedAttention divides the KV Cache into fixed-size physical pages (like OS paging), allowing non-contiguous storage and eliminating internal/external fragmentation.

### Q8: What are the tradeoffs introduced by batching between throughput and latency?
**Answer**: Schedulers increase batch sizes to maximize throughput (tokens per second across all users). However, larger batch sizes increase the time per token (latency) for each individual user, as the GPU takes longer to process larger matrix operations and attention maps.

### Q9: How does queueing delay affect Time-to-First-Token (TTFT) under heavy load?
**Answer**: Under high concurrency, new requests cannot be scheduled immediately because the active batch slots are full. They wait in the queue, which directly increases their TTFT, even though their actual prefill processing time is short.

### Q10: What is P95 latency, and why is it a critical metric in LLM serving?
**Answer**: P95 latency is the latency threshold below which 95% of requests fall. It measures the long-tail behavior, ensuring that even under high concurrency or varying sequence lengths, 95% of users receive an acceptable response time.

### Q11: Why does static batching require padding, and what is its performance cost?
**Answer**: Standard GPU matrix math requires rectangular tensors, so inputs must be padded to the longest sequence in the batch. The GPU performs redundant calculations on these pad tokens, consuming memory and compute bandwidth.

### Q12: How does continuous batching handle the `EOS` (End-of-Sequence) token?
**Answer**: At the end of a forward pass, the scheduler checks if a request generated `EOS`. If yes, its state is marked as finished, its resources (KV cache) are released, and it is removed from the active batch list before the next step.

### Q13: What happens when the KV Cache runs out of memory (OOM) mid-generation?
**Answer**: Schedulers must implement preemptive policies. If VRAM is full, the scheduler must either **preempt** (evict and queue back, discarding generated KV caches) or **swap** (offload KV caches of some requests to CPU RAM) to allow other requests to finish.

### Q14: How does a scheduler decide to prioritize prefill vs. decode requests?
**Answer**: Typically, prefill requests are prioritized to keep TTFT low. Schedulers can group multiple prefills together or run them as soon as slot space opens up.

### Q15: What is "Sequence-Level Batching" vs "Token-Level Batching"?
**Answer**: Sequence-level batching (Static) schedules entire sequences together. Token-level batching (Continuous) schedules token-generation iterations dynamically, allowing sequences to join and leave the execution stream independently.

### Q16: How do you measure GPU utilization in LLM serving, and why is volatile GPU utility misleading?
**Answer**: Volatile GPU utility (`nvidia-smi`) measures the percentage of time a kernel is active. It does not measure compute efficiency. A GPU running batch size 1 can show 99% volatile utility because it is constantly loading weights, while actually performing almost no math. Peak memory allocation and FLOP rate are better indicators.

### Q17: What is the impact of prompt length on prefill latency?
**Answer**: Prefill latency scales quadratically with prompt length due to self-attention computation ($O(N^2)$ matrix multiplications), though it is highly parallelized on the GPU.

### Q18: What is the impact of generation length on decoding latency?
**Answer**: Decoding latency scales linearly ($O(N)$) with output sequence length, since we generate one token per iteration and load model weights at each step.

### Q19: How does continuous batching scale with multi-GPU serving (tensor parallel vs pipeline parallel)?
**Answer**: 
- **Tensor Parallelism**: The scheduler is centralized, and weights are split across GPUs (running parallel matmuls per step).
- **Pipeline Parallelism**: The scheduler must coordinate requests across stages of a pipeline, which is harder because different stages are processing different requests at different times (requiring bubble management).

### Q20: What is the "KV Cache capacity limit"?
**Answer**: It is the maximum number of tokens that can be cached in GPU VRAM after loading the model. It determines the maximum concurrency (batch size) and context length the server can support concurrently without running out of memory.
