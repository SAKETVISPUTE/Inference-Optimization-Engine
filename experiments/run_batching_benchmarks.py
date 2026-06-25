import os
import sys
import time
import json
import torch
import numpy as np

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.loader import load_model_and_tokenizer
from src.scheduler.request import Request
from src.scheduler.scheduler import Scheduler

def generate_dummy_requests(tokenizer, count: int) -> list:
    """Generates a list of dummy requests with varying prompt lengths."""
    prompts = [
        "Explain the theory of relativity in simple terms.",
        "Write a python function to compute the fibonacci sequence.",
        "What are the main causes of climate change?",
        "Translate the following sentence to French: Hello world.",
        "Summarize the plot of the novel Hamlet.",
        "Describe how a CPU processes instructions.",
        "How do you prepare a cup of coffee?",
        "What is the difference between supervised and unsupervised learning?"
    ]
    
    requests = []
    for i in range(count):
        prompt = prompts[i % len(prompts)]
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        # Vary prompt lengths slightly by padding or repeating
        if i >= len(prompts):
            prompt_ids = prompt_ids + [10] * (i % 10)
        requests.append(Request(
            request_id=f"req_{i}",
            prompt=prompt,
            prompt_ids=prompt_ids,
            max_new_tokens=64 + (i * 16) % 128  # varying output lengths
        ))
    return requests

def main():
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. GPU is required for this benchmarking script.")
        sys.exit(1)

    model_id = "Qwen/Qwen2.5-3B-Instruct"
    device = "cuda:0"
    
    print(f"=== Starting Batching Benchmark Suite for {model_id} ===")
    
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_id=model_id, device=device, dtype="float32")
    
    scheduler = Scheduler(model, tokenizer, device=device)
    concurrencies = [1, 2, 4, 8, 16, 32]
    
    results = []
    
    for mode in ["Sequential", "Static", "Continuous"]:
        print(f"\n--- Running Mode: {mode} ---")
        
        for c in concurrencies:
            # Skip massive concurrencies for sequential to keep it fast
            if mode == "Sequential" and c > 8:
                continue
                
            print(f"Testing Concurrency={c}...")
            requests = generate_dummy_requests(tokenizer, c)
            
            # Record start metrics
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            start_time = time.time()
            
            # Execute
            if mode == "Sequential":
                completed = scheduler.run_sequential(requests)
            elif mode == "Static":
                completed = scheduler.run_static_batch(requests)
            elif mode == "Continuous":
                completed = scheduler.run_continuous_batch(requests, max_batch_size=8)
                
            end_time = time.time()
            total_duration = end_time - start_time
            
            # Compute stats
            latencies = []
            total_tokens = 0
            for req in completed:
                if req.finish_time and req.start_time:
                    latencies.append((req.finish_time - req.arrival_time) * 1000.0)
                total_tokens += len(req.generated_ids)
                
            avg_latency = np.mean(latencies) if latencies else 0.0
            p95_latency = np.percentile(latencies, 95) if latencies else 0.0
            throughput = total_tokens / total_duration if total_duration > 0 else 0.0
            gpu_util = torch.cuda.max_memory_allocated(device=device) / (1024 ** 3)  # Peak VRAM as a proxy
            
            record = {
                "mode": mode,
                "concurrency": c,
                "duration_sec": total_duration,
                "total_tokens": total_tokens,
                "throughput_tok_sec": throughput,
                "avg_latency_ms": avg_latency,
                "p95_latency_ms": p95_latency,
                "peak_vram_gb": gpu_util
            }
            results.append(record)
            print(f"  Throughput: {throughput:.2f} tok/s | Avg Latency: {avg_latency:.1f} ms | P95 Latency: {p95_latency:.1f} ms")
            
    # Save results to reports
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/batching_benchmark_results.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nAll batching benchmarks finished! Saved report to {report_path}")

if __name__ == "__main__":
    main()
