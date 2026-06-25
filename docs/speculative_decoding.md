# Speculative Decoding

This document explains the problem, design, acceptance mechanism, implementation, and performance characteristics of our Speculative Decoding engine.

---

## 1. The Problem: Autoregressive Bottleneck

Autoregressive text generation requires executing the model once for every single token generated. Each execution loads the entire set of model weights (e.g., 6GB for a 3B model) from High Bandwidth Memory (HBM) to the GPU SRAM. This is a severe memory-bandwidth bottleneck, limiting execution speeds.

**Speculative Decoding** bypasses this by using a tiny, fast **Draft Model** (e.g., Qwen 2.5-0.5B) to generate a series of candidate tokens (length $K$). A larger, accurate **Verifier Model** (e.g., Qwen 2.5-3B) then verifies all $K$ candidates in a *single parallel forward pass*. Since prompt processing (verification) is compute-bound, it runs almost as fast as generating a single token. If $M$ tokens are accepted, we generate $M+1$ tokens in a single target model step.

---

## 2. Architecture & Design

The execution pipeline functions as follows:

```
           ┌────────────────────────────────────────┐
           │                                        │
Prompt ──► │  Draft Model (0.5B) generates K tokens  │
           │                                        │
           └──────────────────┬─────────────────────┘
                              │ Proposed Tokens
                              ▼
           ┌────────────────────────────────────────┐
           │                                        │
           │  Verifier Model (3B) verifies in      │ ◄─── Target KV Cache
           │  parallel (1 forward pass)             │
           │                                        │
           └──────────────────┬─────────────────────┘
                              │ Logits
                              ▼
           ┌────────────────────────────────────────┐
           │                                        │
           │  Acceptance Decision & Cache Rollback   │ ──► Final Output
           │                                        │
           └────────────────────────────────────────┘
```

- **[draft_model.py](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/src/speculative/draft_model.py)**: Wraps the small draft model and manages its KV Cache to propose $K$ candidate tokens autoregressively.
- **[verifier.py](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/src/speculative/verifier.py)**: Wraps the target model to execute parallel verification over the concatenated input.
- **[acceptance.py](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/src/speculative/acceptance.py)**: Implements greedy acceptance, tracking where predictions match and truncating/rolling back both target and draft KV caches upon rejection.
- **[speculative_generator.py](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/src/speculative/speculative_generator.py)**: Orchestrates the draft, verify, and rollback loop.

---

## 3. Benchmark Results & Analysis

Quantitative comparison of Standard (3B model) vs. Speculative Decoding (3B verifier + 0.5B draft) on an NVIDIA RTX A6000 GPU (48GB VRAM) using **Qwen 2.5-3B** as target and **Qwen 2.5-0.5B** as draft:

### Performance Comparison Table

| Output Length | Standard Speed (tok/s) | Speculative Speed (tok/s) | Speedup (x) | Acceptance Rate (%) |
|---|---|---|---|---|
| **32** | 32.37 | 17.65 | 0.55x | 23.4% |
| **128** | 32.90 | 24.34 | 0.74x | 41.7% |
| **512** | 32.16 | 21.56 | 0.67x | 35.8% |
| **1024** | 32.79 | 22.95 | 0.70x | 38.6% |

---

### Result Analysis: The Speculative Tradeoff

The benchmark results show a speedup of **0.55x to 0.74x** (a slowdown). This is a well-documented behavior in native PyTorch implementations of speculative decoding. Here is the deep technical explanation of why this occurs and how to unlock actual speedup:

#### 1. Why Acceptance Rate Matters
The acceptance rate indicates the accuracy of the draft model relative to the target model. Our observed rate hovered around **35% - 42%**.
- For every speculative step of $K=4$ proposed tokens, the verifier model rejected draft tokens after the 1st or 2nd token on average.
- When a token is rejected, the target model must discard the rest of the draft, roll back the KV caches, and perform another step. If we accept only 1 token per verification, we have paid the target model forward pass cost plus the draft model overhead, but generated only 2 tokens. This is slower than generating 2 tokens sequentially with the target model.

#### 2. The Draft Model Loop Overhead
The draft model (`0.5B`) is executed autoregressively in a Python loop for $K$ steps. 
- Because the draft model is small, its GPU computation time is extremely short. However, launching each GPU kernel from Python introduces a CPU-to-GPU overhead (approx. 0.1-0.2ms per kernel).
- Doing this $K$ times sequentially in a Python loop creates a cumulative **Draft Overhead** that exceeds the time saved by the verifier's single forward pass.

#### 3. How to Unlock Real Speedup (Production Optimizations)
To move from `0.7x` to `2.0x` speedup, production engines implement:
1. **Draft Kernel Fusion / Compilation**: Compiling the draft model (e.g. using `torch.compile` or CUDA graph capture) to eliminate Python interpreter and kernel launch overhead.
2. **Tree-based Verification**: Instead of verifying a single linear sequence of candidate tokens, the draft model generates a tree of possible paths. The verifier checks all paths in the tree in a single pass, increasing the effective acceptance rate to 70-80%.
3. **Model Size Alignment**: Ensuring the target model is significantly larger (e.g., 70B parameters) relative to the draft (e.g., 1.5B or 7B). If the target is massive, its weight loading time is huge, making the draft model's overhead relatively negligible and unlocking massive speedups.


---

## 4. Interview Prep: 20 Speculative Decoding Questions

### Q1: What is the core speedup mechanism in speculative decoding?
**Answer**: Autoregressive decoding is memory-bandwidth bound. We generate candidate tokens with a tiny model (which loads small weights quickly), and verify them in a single parallel step with the large model. Verification is compute-bound, meaning the verifier can inspect $K$ tokens in nearly the same time it would take to generate just 1 token.

### Q2: Why is verification of K tokens in the target model faster than generating K tokens autoregressively?
**Answer**: Verification processes the $K$ tokens in parallel in a single forward pass, which has high arithmetic intensity and utilizes the GPU's Tensor Cores (compute-bound). Autoregressive generation requires $K$ sequential passes, each of which must read all target weights from HBM to SRAM (memory-bandwidth bound).

### Q3: What is the "Acceptance Rate", and why is it the key driver of speculative speedup?
**Answer**: The acceptance rate is the fraction of draft-proposed tokens that match the verifier's choices. A higher acceptance rate means more tokens are verified per target forward pass, leading to higher speedup. If the rate is 0%, we get no speedup (and instead suffer draft model overhead).

### Q4: Explain the mathematical expectation of speedup in speculative decoding.
**Answer**: Expected speedup depends on the average number of accepted tokens $\alpha$. If the average number of accepted tokens per step is $\alpha$, we generate $\alpha + 1$ tokens per target model step. The speedup is approximately $\frac{\alpha + 1}{1 + c \cdot (K)}$ where $c$ is the ratio of draft model latency to target model latency.

### Q5: How do we handle KV Cache rollback when a proposed token is rejected?
**Answer**: If the verifier rejects the draft token at index $i$, all subsequent proposed tokens are discarded. We must truncate the KV caches of both the target and draft models to match the accepted sequence length (removing keys and values of rejected tokens) so that the next step starts from the correct state.

### Q6: Why is the choice of draft model size critical?
**Answer**: 
- If the draft model is **too small**: It is very fast, but its output quality is poor, leading to a low acceptance rate and low speedup.
- If the draft model is **too large**: Its acceptance rate is high, but it takes too long to generate the draft tokens, eating into the time saved by the verifier.
The optimal draft model is usually 5x to 10x smaller than the target model.

### Q7: Can speculative decoding yield outputs that differ from the target model's standard generation?
**Answer**: No. Under greedy decoding, verification is mathematically identical to standard target generation because we verify against the target's argmax logits and immediately correct any mismatch. Under nucleus/temperature sampling, modified acceptance algorithms (like GETH or stochastic verification) guarantee that the output distribution matches the target model's distribution exactly.

### Q8: What is the optimal number of draft tokens K to propose?
**Answer**: It is a tradeoff. Proposing more tokens (larger $K$) increases the potential speedup if the draft model is highly accurate, but increases draft latency and verification compute if tokens are rejected early. Usually, $K \in [3, 6]$ is optimal.

### Q9: How does prompt prefix caching interact with speculative decoding?
**Answer**: Prompt prefix caching caches the KV states of system prompts or common prefixes. This speeds up the prefill phase of both the draft and target models, but does not affect the subsequent speculative decoding steps.

### Q10: How does speculative decoding perform on coding vs. creative writing tasks?
**Answer**: Speculative decoding generally gets higher speedups on highly structured or predictable text (like code or formal reasoning) because the draft model is more likely to match the target. It gets lower speedups on highly creative or open-ended tasks where the token probability distribution is flatter.

### Q11: Explain the GETH (Generalized Expected Token Hypothesis) or standard rejection sampling in speculative decoding.
**Answer**: For sampling (non-greedy), a draft token $x$ is accepted with probability $\min(1, \frac{P(x)}{Q(x)})$ where $P$ is target probability and $Q$ is draft probability. If rejected, we sample a new token from the normalized difference distribution $\max(0, P(x) - Q(x))$. This guarantees output parity with the target distribution.

### Q12: Why does speculative decoding not work well under high batch sizes (high concurrency)?
**Answer**: Under high batch sizes, the target model's forward pass becomes compute-bound rather than memory-bound (since we are doing math for many sequences). Since the verifier is already fully utilizing the GPU compute, the draft model overhead (which is also batched) yields diminishing speedup or even slows down generation.

### Q13: What is "Medusa" or "Blockwise Parallel Decoding", and how does it compare to speculative decoding?
**Answer**: Medusa does not use a separate draft model. Instead, it adds multiple extra heads to the target model, where head $i$ predicts the token $i$ steps in the future. The target model verifies its own multiple heads' predictions in a single pass. It eliminates the need to run and manage a separate draft model.

### Q14: How does speculative decoding scale with hardware (e.g. slower HBM vs faster Tensor Cores)?
**Answer**: Speculative decoding yields higher speedups on systems with a larger gap between memory bandwidth and compute performance (e.g., edge devices, CPUs, or older GPUs) because the memory weight-loading bottleneck is more severe.

### Q15: What is the "Draft Overhead"?
**Answer**: It is the cumulative time spent running the draft model's autoregressive loop. If $K=4$, we run the draft model 4 times per verification step. This overhead is paid at every step, regardless of how many tokens are accepted.

### Q16: How do you verify that the output of speculative decoding is correct?
**Answer**: We compare the output text and generated token IDs of speculative decoding with standard greedy generation from the target model. Under greedy decoding, they must match token-for-token (100% parity).

### Q17: What are the main memory challenges when running speculative decoding?
**Answer**: We must load both the target model and the draft model weights into GPU memory, and manage two separate sets of KV caches. This increases the VRAM footprint and reduces the VRAM available for user batches.

### Q18: What is "Staged Speculative Decoding"?
**Answer**: Staged speculative decoding uses a tree-based draft generation. Instead of generating a single chain, it generates a tree of possible sequences from the draft, which the verifier checks in parallel. This increases the chance of finding an acceptable path, raising the effective acceptance rate.

### Q19: Explain the impact of quantization (e.g., INT8/INT4) on speculative decoding.
**Answer**: Quantizing the target model reduces its size and memory loading time, speeding up standard generation. However, it also reduces target forward pass latency, which can reduce the relative speedup ratio of speculative decoding unless the draft model is also optimized/quantized.

### Q20: What is the "rejection index"?
**Answer**: The rejection index is the index $i \in [0, K-1]$ of the first draft token that is rejected by the verifier. All tokens from index $i$ to $K-1$ are discarded, and the verifier's predicted token at position $i$ is used as the correction token.
