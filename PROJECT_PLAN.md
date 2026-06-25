# High Performance LLM Inference Engine: Project Plan

This document outlines the design, architecture, implementation roadmap, and validation strategies for our custom LLM inference engine. The goal of this project is to build a lightweight, highly-optimized, and well-understood serving engine, quantitatively demonstrating the impact of modern inference optimizations.

---

## 1. Project Overview & Objectives
We are building a custom inference engine using **Qwen 2.5-3B Instruct** (or Llama 3.2-3B) as our target model. 
By designing and implementing critical components from scratch, we aim to answer:
> *"Which inference optimizations matter most, under which workloads, and why?"*

### Performance Metrics Focus
* **TTFT (Time to First Token)** (ms)
* **TPOT (Time Per Output Token)** or **Tokens/sec** (token/s)
* **Total Latency** (ms)
* **Peak VRAM Consumption** (GB)
* **Throughput** (tokens/sec across multiple requests)

---

## 2. Directory Layout
We will organize our codebase inside the workspace as follows:

```
/mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/
├── PROJECT_PLAN.md
├── TODO.md
├── src/
│   ├── decoder/         # Custom Autoregressive Generation loop
│   ├── cache/           # Key-Value Cache implementation
│   ├── quantization/    # INT8 & INT4 Quantization logic
│   ├── benchmark/       # Benchmarking framework
│   ├── profiling/      # torch.profiler and GPU memory tracking utilities
│   ├── models/          # Model loading and wrapper code (Qwen/Llama)
│   └── utils/           # Helper scripts (logging, config loader, etc.)
├── experiments/         # Scripts for running specific test scenarios
├── plots/               # Performance visual graphs (SVG/PNG)
├── reports/             # Aggregated benchmark run results (JSON/CSV)
├── tests/               # Unit and regression tests
├── configs/             # Configuration files for experiments/models
├── docs/                # Concise, markdown-based documentation
│   ├── engineering_journal.md
│   ├── generation.md
│   ├── kv_cache.md
│   ├── int8_quantization.md
│   └── int4_quantization.md
└── future/              # Future feature design documents
    ├── continuous_batching_design.md
    ├── speculative_decoding_design.md
    └── prefix_cache_design.md
```

---

## 3. Milestones & Timeline

### Phase 1: Custom Autoregressive Generation Engine
* **Goal**: Implement a custom generation loop that loads model weights, tokenizes inputs, generates token-by-token using greedy decoding, and supports basic sampling (temperature, top-p/top-k).
* **Deliverable**: A working generation script.
* **Validation**: Output parity with Hugging Face standard `model.generate()`.
* **Documentation**: `docs/generation.md`.

### Phase 2: Key-Value (KV) Cache
* **Goal**: Modify the attention layer/forward pass to cache Key and Value tensors. Eliminate the redundant $O(N^2)$ computation for past tokens during generation, changing sequence-level decoding complexity from $O(N^2)$ to $O(N)$.
* **Deliverable**: KV cache integration into the forward pass.
* **Validation**: Parity in output logits/text compared to Phase 1.
* **Documentation**: `docs/kv_cache.md` + initial benchmarks.

### Phase 3: INT8 Quantization
* **Goal**: Implement INT8 quantization (e.g., symmetric weight-only linear layers or activation/weight quantization) to reduce VRAM footprint.
* **Deliverable**: Quantized model weight loading and custom quantized matmul kernel wrappers.
* **Validation**: Text generation sanity check and relative perplexity/accuracy comparison.
* **Documentation**: `docs/int8_quantization.md`.

### Phase 4: INT4 Quantization
* **Goal**: Implement INT4 quantization (e.g., weight-only group-wise quantization) to further compress the model memory footprint.
* **Deliverable**: INT4 compression code and weight-unpacking/dequantization steps.
* **Validation**: Generation sanity checks.
* **Documentation**: `docs/int4_quantization.md`.

### Phase 5: Comprehensive Benchmark & Profiling Framework
* **Goal**: Create a reusable test-bench targeting varying context lengths (512 to 16,384) and output lengths (32 to 1024). Run profiling using PyTorch Profiler (`torch.profiler`) and track VRAM peak limits.
* **Deliverable**: Profiling scripts, tables, plots, and analysis reports in `reports/` and `plots/`.
* **Documentation**: `docs/benchmarking.md` and `docs/profiling.md`.

### Phase 6: Final Results Analysis & Synthesis
* **Goal**: Consolidate findings into a final analytical report. Write the continuous engineering journal.
* **Deliverable**: `docs/engineering_journal.md`.

### Phase 7: Continuous Batching Scheduler
* **Goal**: Implement request queuing, active batch slot management, and dynamic token-level scheduling to maximize serving throughput.
* **Deliverable**: Request queue, batch manager, scheduler execution framework.
* **Documentation**: `docs/continuous_batching.md`.

### Phase 8: Speculative Decoding
* **Goal**: Implement draft generation, parallel verification, acceptance/rejection checking, and target/draft KV cache rollback to accelerate autoregressive generation.
* **Deliverable**: Draft model wrapper, verifier model wrapper, acceptance checker, speculative generation pipeline.
* **Documentation**: `docs/speculative_decoding.md`.

---

## 4. Hardware and Technical Configuration
* **Hardware**: Single GPU with ~49GB VRAM.
* **Software**: PyTorch, Hugging Face Transformers, Accelerator, tokenizers.
* **Target Model**: `Qwen/Qwen2.5-3B-Instruct`
* **Draft Model**: `Qwen/Qwen2.5-0.5B-Instruct`

---

## 5. Architectural Considerations for Future Extensions (Phase 9)
We will design interfaces (e.g., decoupled forward passes and cache managers) that will seamlessly support:
1. **Prefix Caching**: Sharing KV caches of system prompts across multiple client sessions.
2. **Chunked Prefill**: Merging long prefill prompts and single decode tokens into a single execution step to avoid blocking decodes.

