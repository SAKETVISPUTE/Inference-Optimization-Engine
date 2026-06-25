# High Performance LLM Inference Engine: TODO List

## Phase 0: Setup
- [x] Create directory structure
- [x] Create `PROJECT_PLAN.md`
- [x] Create `TODO.md`

## Phase 1: Custom Autoregressive Generation Engine
- [x] Create wrapper script to download and load `Qwen/Qwen2.5-3B-Instruct`
- [x] Implement tokenizer encoding/decoding wrapper
- [x] Implement greedy token generation logic
- [x] Implement sampling token generation logic (temperature, top-p, top-k)
- [x] Validate generation correctness against Hugging Face default implementation
- [x] Create `docs/generation.md`
- [x] Log Phase 1 in `docs/engineering_journal.md`

## Phase 2: Key-Value (KV) Cache
- [x] Design cache management system
- [x] Modify model attention layers to accept/return and update cached Key and Value tensors
- [x] Run autoregressive generation with KV Cache enabled
- [x] Validate correctness (logits match baseline without cache)
- [x] Benchmark KV Cache vs. No-Cache baseline (TTFT, TPOT, VRAM, Throughput)
- [x] Create `docs/kv_cache.md`
- [x] Log Phase 2 in `docs/engineering_journal.md`

## Phase 3: INT8 Quantization
- [x] Implement INT8 quantization calibration / weight conversion
- [x] Implement custom forward pass wrapper with INT8 matmuls
- [x] Validate quantized model output sanity and generation correctness
- [x] Benchmark INT8 quantized vs. FP16 baseline
- [x] Create `docs/int8_quantization.md`
- [x] Log Phase 3 in `docs/engineering_journal.md`

## Phase 4: INT4 Quantization
- [x] Implement INT4 quantization compression/packing algorithm
- [x] Implement weight unpack/dequantization and customized linear layer replacement
- [x] Validate quantized model output sanity and correctness
- [x] Benchmark INT4 quantized vs. INT8/FP16 baselines
- [x] Create `docs/int4_quantization.md`
- [x] Log Phase 4 in `docs/engineering_journal.md`

## Phase 5: Benchmark & Profiling Framework
- [x] Implement `torch.profiler` profiling utilities
- [x] Implement peak VRAM and GPU utilization tracking code
- [x] Run benchmark suite across defined ranges
- [x] Generate tables, comparison plots, and report files in `reports/` and `plots/`
- [x] Create `docs/benchmarking.md` and `docs/profiling.md`
- [x] Log Phase 5 in `docs/engineering_journal.md`

## Phase 6: Final Report and Archiving
- [ ] Synthesize all benchmark findings
- [ ] Finalize `docs/engineering_journal.md`

## Phase 7: Continuous Batching Scheduler
- [x] Explain Continuous Batching systems concept and throughput-latency tradeoffs
- [x] Create `src/scheduler/` directory
- [x] Implement Request, RequestQueue, BatchManager, and iteration-level Scheduler
- [x] Add benchmarking script `experiments/run_batching_benchmarks.py`
- [x] Execute benchmarks comparing Sequential, Static Batching, and Continuous Batching
- [x] Document architecture and interview questions in `docs/continuous_batching.md`

## Phase 8: Speculative Decoding
- [x] Explain Speculative Decoding autoregressive speedup concept
- [x] Create `src/speculative/` directory
- [x] Implement DraftModelWrapper, VerifierModelWrapper, greedy acceptance logic, and KV cache rollback
- [x] Add benchmarking script `experiments/run_speculative_benchmarks.py`
- [x] Execute benchmarks comparing Standard target generation with Speculative Decoding
- [x] Document architecture and interview questions in `docs/speculative_decoding.md`

