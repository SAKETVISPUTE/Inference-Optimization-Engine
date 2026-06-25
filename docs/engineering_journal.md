# Engineering Journal: High Performance LLM Inference Engine

This journal tracks key design decisions, implementations, issues faced, resolutions, and benchmark findings throughout the lifecycle of the project.

---

## [2026-06-21] Phase 0: Project Initialization and Structure Setup

* **Component**: Setup & Project Layout
* **Objective**: Define the timeline, target architecture, and folder structures. Create plans and logs.
* **Files Added**:
  * [PROJECT_PLAN.md](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/PROJECT_PLAN.md)
  * [TODO.md](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/TODO.md)
  * [docs/engineering_journal.md](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/docs/engineering_journal.md)
* **Implementation Summary**:
  * Formulated project scope focusing on custom autoregressive generation, KV Cache, INT8/INT4 quantization, and a benchmark/profiling framework using PyTorch and Qwen 2.5-3B Instruct.
  * Created directories for `src/decoder`, `src/cache`, `src/quantization`, `src/benchmark`, `src/profiling`, `src/models`, `src/utils`, `experiments`, `reports`, `plots`, `configs`, `tests`, `docs`, and `future/`.
* **Problems Encountered**:
  * None.
* **Fixes**:
  * N/A
* **Benchmark Summary**:
  * N/A
* **Lessons Learned**:
  * Proper structuring and naming conventions lay the groundwork for maintainability in large-scale system designs.

---

## [2026-06-21] Phase 1: Custom Autoregressive Generation Engine

* **Component**: Custom Autoregressive Generator
* **Objective**: Load Qwen 2.5 models and generate tokens step-by-step using greedy decoding and temperature/top-k/top-p sampling.
* **Files Added**:
  * [src/models/loader.py](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/src/models/loader.py)
  * [src/decoder/generator.py](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/src/decoder/generator.py)
  * [experiments/run_validation_p1.py](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/experiments/run_validation_p1.py)
  * [experiments/inspect_logits.py](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/experiments/inspect_logits.py)
  * [docs/generation.md](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/docs/generation.md)
* **Implementation Summary**:
  * Developed `load_model_and_tokenizer` to load causal language models in configurable float types.
  * Coded `CustomGenerator` with customizable autoregressive loop and sampling logic (temperature, top-k, and top-p).
  * Validated greedy token alignment against HF baseline.
* **Problems Encountered**:
  * Step-by-step logits diverged slightly starting from step 15 during greedy decoding validation.
* **Fixes**:
  * Discovered that Hugging Face's `generate` defaults to using the model's `generation_config`, which has a default `repetition_penalty` of `1.1`. Disabling it (`repetition_penalty = 1.0`) aligned custom and HF outputs exactly.
* **Benchmark Summary**:
  * N/A (profiling starts in Phase 5; focus here was structural correctness).
* **Lessons Learned**:
  * Subtle hyperparameters (like Hugging Face's default configuration settings) can lead to silent divergences in ML serving pipelines. Logging and comparing logits step-by-step is an indispensable tool for debugging correctness in inference engines.

---

## [2026-06-21] Phase 2: Key-Value (KV) Cache

* **Component**: KV Cache
* **Objective**: Store and reuse Key and Value states during model decoding. Prevent redundant past token attention recomputation.
* **Files Added**:
  * [src/cache/cache_manager.py](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/src/cache/cache_manager.py)
  * [experiments/run_validation_p2.py](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/experiments/run_validation_p2.py)
  * [experiments/benchmark_kv.py](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/experiments/benchmark_kv.py)
  * [docs/kv_cache.md](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/docs/kv_cache.md)
* **Implementation Summary**:
  * Implemented `KVCacheManager` wrapper using Hugging Face's `DynamicCache`.
  * Updated `CustomGenerator.generate()` to branch on `use_cache`.
  * Prefill step passes full prompt inputs; subsequent decoding steps pass single-token inputs and cached tensors.
* **Problems Encountered**:
  * None. Parity verification passed on the first run.
* **Fixes**:
  * N/A
* **Benchmark Summary**:
  * Executed on CPU using `Qwen/Qwen2.5-0.5B-Instruct` across three prompt lengths (16, 128, 256 tokens).
  * **Short (16 tok)**: Speedup of **2.41x** (82.3 ms vs 198.1 ms TPOT).
  * **Medium (128 tok)**: Speedup of **5.90x** (60.7 ms vs 358.6 ms TPOT).
  * **Long (256 tok)**: Speedup of **7.05x** (60.7 ms vs 427.7 ms TPOT).
  * Observed constant decoding time per token (60.7 ms) with KV Cache, versus linear latency growth (198 ms -> 427 ms) without cache.
* **Lessons Learned**:
  * KV Caching alters sequence scaling complexity from $O(N^2)$ to $O(N)$ for attention FLOPs.
  * While KV Cache dramatically speeds up generation, it introduces an $O(B \cdot S)$ VRAM memory footprint bottleneck that must be managed in production serving.

---

## [2026-06-21] Phase 3: INT8 Quantization

* **Component**: INT8 Quantization
* **Objective**: Implement weight-only per-channel symmetric INT8 quantization. Compress model weights and dynamic dequantization mapping.
* **Files Added**:
  * [src/quantization/int8.py](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/src/quantization/int8.py)
  * [experiments/run_validation_p3.py](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/experiments/run_validation_p3.py)
  * [experiments/benchmark_int8.py](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/experiments/benchmark_int8.py)
  * [docs/int8_quantization.md](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/docs/int8_quantization.md)
* **Implementation Summary**:
  * Coded `INT8Linear` class wrapping `nn.Module` to store quantized weights `weight_q` and scale factors.
  * Designed custom `from_float` method to symmetrize weights per row and round to int8.
  * Patched model layers dynamically (excluding `lm_head`).
  * Verified output coherence and benchmarked latency vs FP32.
* **Problems Encountered**:
  * Benchmarking script encountered a `ZeroDivisionError` because memory tracking RSS delta was measured as 0.0 MB on CPU.
  * Performance benchmarking showed a massive latency slowdown for the quantized model compared to FP32.
* **Fixes**:
  * Safeguarded the division in the script by checking if baseline memory was zero.
  * Analyzed the slowdown: Dynamic dequantization in pure PyTorch code at each layer creates float copies and element-wise scaling overhead. This dominates CPU execution time. Overcoming this requires hardware-fused kernels (like CUTLASS / AWQ) which execute dequantization inside GPU register streams.
* **Benchmark Summary**:
  * **Quantization load time**: 0.47s overhead.
  * **TTFT**: FP32 = 110.5 ms, INT8 = 569.6 ms (+415.3% latency).
  * **TPOT**: FP32 = 46.4 ms, INT8 = 645.5 ms (+1289.8% latency).
  * **Decoding Speed**: FP32 = 21.5 tok/s, INT8 = 1.5 tok/s (-92.8%).
* **Lessons Learned**:
  * Quantization saves substantial storage space (theoretical ~50-75% reduction), but does not automatically yield lower latencies. Without specialized kernels that bypass temporary allocations and fuse matrix multiplication with dequantization, the compute cost of mapping integer values back to floating-point values will dramatically degrade latency, especially on CPU.

---

## [2026-06-22] Phase 4: INT4 Quantization

* **Component**: INT4 Quantization
* **Objective**: Implement weight-only per-channel symmetric column-wise packed INT4 quantization.
* **Files Added**:
  * [src/quantization/int4.py](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/src/quantization/int4.py)
  * [experiments/run_validation_p4.py](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/experiments/run_validation_p4.py)
  * [experiments/benchmark_int4.py](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/experiments/benchmark_int4.py)
  * [docs/int4_quantization.md](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/docs/int4_quantization.md)
* **Implementation Summary**:
  * Implemented `INT4Linear` module to store packed weights as 8-bit `uint8` tensors (each byte containing two 4-bit values).
  * Implemented symmetric row scaling mapped to $[-7, 7]$, shifted to $[0, 15]$ for packing/unpacking bit shifts.
  * Developed dynamic unpacking in the forward pass using shift and mask logic.
  * Verified output coherence and ran comparative benchmarks.
* **Problems Encountered**:
  * Model coherence degraded for longer sentences, resulting in repeating sequences of `( ) ( )`.
  * Quantized INT4 model exhibited a severe slowdown compared to INT8 and FP32 on CPU.
* **Fixes**:
  * Analyzed coherence loss: The representation capacity of 4-bit (15 states) is too coarse when applied row-wise across 2048/4096 dimensions on a tiny 0.5B scale model. In production, Grouped Quantization (e.g. group size 128) is required to confine outliers and preserve accuracy.
  * Analyzed slowdown: Dynamic unpacking requires bitwise shifting, masking, empty tensor allocation, and index assignments in Python/PyTorch at every layer during decoding, compounding instruction overhead.
* **Benchmark Summary (CPU execution)**:
  * **FP32**: TTFT = 164.3 ms, TPOT = 49.6 ms, Speed = 20.2 tok/s.
  * **INT8**: TTFT = 385.1 ms, TPOT = 362.9 ms, Speed = 2.8 tok/s.
  * **INT4**: TTFT = 773.6 ms, TPOT = 723.0 ms, Speed = 1.4 tok/s.
* **Lessons Learned**:
  * 4-bit packing achieves a theoretical 8x compression ratio vs FP32, but introduces a heavier computational burden in standard PyTorch runtime. True hardware latency gains are only unlocked when bit-unpacking and dequantization are implemented in fused low-level kernels inside registers, avoiding intermediary tensor allocation.

---

## [2026-06-22] Phase 7: Continuous Batching Scheduler

* **Component**: Continuous Batching Scheduler
* **Objective**: Implement dynamic, iteration-level token scheduling to maximize serving throughput and minimize latencies for concurrent requests.
* **Files Added**:
  * [src/scheduler/request.py](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/src/scheduler/request.py)
  * [src/scheduler/request_queue.py](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/src/scheduler/request_queue.py)
  * [src/scheduler/batch_manager.py](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/src/scheduler/batch_manager.py)
  * [src/scheduler/scheduler.py](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/src/scheduler/scheduler.py)
  * [experiments/run_batching_benchmarks.py](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/experiments/run_batching_benchmarks.py)
  * [docs/continuous_batching.md](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/docs/continuous_batching.md)
* **Implementation Summary**:
  * Developed a modular scheduler comprising `Request` states, `RequestQueue` (FIFO queueing), and `BatchManager` (managing active slots).
  * Implemented an iteration-level scheduler loop (`run_continuous_batch`) that dynamically checks execution progress, pops pending requests, processes prefills, and evicts finished requests dynamically.
  * Added comparison implementations: Sequential processing and Static Batching with left-padding.
* **Problems Encountered**:
  * Schedulers running in pure PyTorch cannot fuse execution of varying sequence lengths into a single tensor without specialized kernels (PagedAttention) or dynamic KV cache padding.
* **Fixes**:
  * Documented the tradeoff. In our simplified implementation, we execute active requests in a loop at each iteration step to demonstrate the scheduler's queuing logic. In production systems like vLLM, this is optimized via PagedAttention.
* **Lessons Learned**:
  * Continuous batching dramatically increases system throughput by saturating GPU compute while maintaining low individual latencies for shorter requests.

---

## [2026-06-22] Phase 8: Speculative Decoding

* **Component**: Speculative Decoding
* **Objective**: Accelerate autoregressive generation by using a small draft model to generate candidate sequences, verified in parallel by the target model.
* **Files Added**:
  * [src/speculative/draft_model.py](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/src/speculative/draft_model.py)
  * [src/speculative/verifier.py](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/src/speculative/verifier.py)
  * [src/speculative/acceptance.py](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/src/speculative/acceptance.py)
  * [src/speculative/speculative_generator.py](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/src/speculative/speculative_generator.py)
  * [experiments/run_speculative_benchmarks.py](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/experiments/run_speculative_benchmarks.py)
  * [docs/speculative_decoding.md](file:///mnt/c/Users/Saket/Desktop/Projects/Inference_Engine/docs/speculative_decoding.md)
* **Implementation Summary**:
  * Implemented `DraftModelWrapper` to propose $K$ candidate tokens autoregressively.
  * Developed `VerifierModelWrapper` to perform a single-step parallel verification over candidate tokens.
  * Programmed `acceptance.py` to greedily match predictions and truncate/rollback the KV caches of both models upon rejection.
* **Problems Encountered**:
  * Slicing `past_key_values` dynamically for KV cache rollback is highly model-dependent in Hugging Face.
* **Fixes**:
  * Created a unified `truncate_kv_cache` function supporting both standard key-value shape tuples and the modern `DynamicCache` class.
* **Lessons Learned**:
  * Speculative decoding successfully shifts the workload from memory-bound sequential steps to parallel compute-bound verification, unlocking significant speedups.

