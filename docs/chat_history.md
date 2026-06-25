# Conversation History

This file maintains a high-level summary of our conversations, decisions, and alignments throughout the project, enabling us to continue seamlessly across sessions.

---

## Session 1: 2026-06-21

### User Requests
* Act as a Technical Lead, Systems Architect, ML Systems Mentor, and Performance Engineering Mentor.
* Build a High Performance LLM Inference Engine. Focus on custom autoregressive generation, KV Cache, INT8/INT4 quantization, and a benchmark/profiling framework.
* Target model: `Qwen/Qwen2.5-3B-Instruct` (using `Qwen/Qwen2.5-0.5B-Instruct` for local CPU testing).
* Implement Phase 0 (Structure, Plans) and Phase 1 (Custom Autoregressive Generation).
* Maintain a chat history file to continue conversations.

### Key Decisions & Alignments
1. **Target Model Selection**: Chose `Qwen2.5` because it has open weights (avoiding gated model tokens like Llama 3.2). Used `Qwen/Qwen2.5-0.5B-Instruct` for rapid local CPU testing.
2. **GPU Connectivity**: The user clarified that the GPU will be connected later via SSH, meaning we start development and validation on the local CPU workspace.
3. **Parity Check**: Custom generation must match Hugging Face's built-in `generate()` token-for-token when run in greedy mode.

### Status Summary
* **Phase 0**: Completed. Created `PROJECT_PLAN.md`, `TODO.md`, `docs/engineering_journal.md`, and the folder structures.
* **Phase 1**: Completed. Custom autoregressive generation engine implemented, validated (100% token parity with HF greedy decoding on Qwen2.5), and documented in `docs/generation.md`.
* **Phase 2**: Completed. KV Cache implemented, mathematically validated (100% token parity with and without cache), benchmarked (up to 7.05x speedup on CPU), and documented in `docs/kv_cache.md`.
* **Phase 3**: Completed. INT8 weight-only quantization implemented, validated for coherence, benchmarked (measured dynamic dequantization cost on CPU), and documented in `docs/int8_quantization.md`.
* **Phase 4**: Completed. INT4 weight-only packed quantization (column-wise 4-bit packing) implemented, validated (Qwen2.5 generates correct short sentences), benchmarked (measured unpacking instruction overhead on CPU), and documented in `docs/int4_quantization.md`.

---

## Session 2: 2026-06-22

### User Requests
* Resume Phase 5 (Benchmark & Profiling Framework) on GPUs via SSH.
* If tmux session was killed on the server, create a new session and start benchmarks.
* Work must be confined to the folder `/home/vlmdg/workspace/Saket` on the remote server.

### Key Decisions & Alignments
1. **VPN/SSH Connection Conflict**: Connection to the GPU host `10.107.105.52:2225` requires the user's VPN to be active. However, when the VPN was active, the agent's connection to the Google Gemini API server got blocked. The user resolved this by enabling VPN split-tunneling.
2. **MTU Mismatch Fix**: After VPN activation, the SSH connection hung on key exchange (`expecting SSH2_MSG_KEX_ECDH_REPLY`). The user resolved this by lowering the WSL MTU to 1300 (`sudo ip link set dev eth0 mtu 1300`).
3. **Benchmark Script Hotfix**: Fixed missing quantization imports (`quantize_model_to_int8` and `quantize_model_to_int4`) in `experiments/run_gpu_benchmarks.py` both locally and on the remote host to avoid crashes during execution.

### Status Summary
* **Phase 5 Benchmarks & Profiling**: Completed. The remote run completed successfully. Checked results and copied the `gpu_benchmark_results.json` locally.
* **Phase 7 Continuous Batching**: Completed. Implemented the Request queue, BatchManager, and Scheduler (`run_continuous_batch`) in `src/scheduler/`. Executed remote benchmarks and recorded the results (e.g. 520+ tok/s throughput on A6000 under batch size 32). Documented design and 20 interview questions in `docs/continuous_batching.md`.
* **Phase 8 Speculative Decoding**: Completed. Implemented the Draft Model, Verifier, acceptance logic, and KV cache rollback in `src/speculative/`. Run speculative generation benchmarks using Qwen2.5-3B (Target) and Qwen2.5-0.5B (Draft). Analyzed the results (e.g., draft loop overhead vs. acceptance rate) and documented findings along with 20 interview questions in `docs/speculative_decoding.md`.







