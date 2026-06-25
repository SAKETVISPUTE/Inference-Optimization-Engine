import os
import sys
import time
import json
import torch

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.loader import load_model_and_tokenizer
from src.speculative.speculative_generator import SpeculativeGenerator
from src.decoder.generator import CustomGenerator

def main():
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. GPU is required for this benchmarking script.")
        sys.exit(1)

    target_id = "Qwen/Qwen2.5-3B-Instruct"
    draft_id = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "cuda:0"
    
    print(f"=== Starting Speculative Decoding Benchmark Suite ===")
    
    print("Loading target model...")
    target_model, tokenizer = load_model_and_tokenizer(model_id=target_id, device=device, dtype="float32")
    
    print("Loading draft model...")
    draft_model, _ = load_model_and_tokenizer(model_id=draft_id, device=device, dtype="float32")
    
    spec_gen = SpeculativeGenerator(target_model, draft_model, tokenizer, device=device)
    std_gen = CustomGenerator(target_model, tokenizer)
    
    prompt = "Explain quantum superposition in detail, including its mathematical representation and physical significance."
    output_lens = [32, 128, 512, 1024]
    
    results = []
    
    for out_len in output_lens:
        print(f"\n--- Testing Output Length: {out_len} ---")
        
        # 1. Run standard target model generation
        print("Running standard target generation...")
        torch.cuda.empty_cache()
        std_start = time.time()
        # Warmup
        _ = std_gen.generate(prompt=prompt, max_new_tokens=4, use_cache=True)
        
        std_start_actual = time.time()
        std_output = std_gen.generate(prompt=prompt, max_new_tokens=out_len, use_cache=True)
        std_time = time.time() - std_start_actual
        std_tokens = len(std_output["generated_tokens"])
        std_speed = std_tokens / std_time if std_time > 0 else 0.0
        print(f"  Standard Speed: {std_speed:.2f} tok/s | Latency: {std_time * 1000.0:.1f} ms")
        
        # 2. Run speculative decoding
        print("Running speculative generation...")
        torch.cuda.empty_cache()
        # Warmup
        _, _, _ = spec_gen.generate(prompt=prompt, max_new_tokens=4, k=4)
        
        spec_start = time.time()
        spec_output, generated_ids, spec_metrics = spec_gen.generate(prompt=prompt, max_new_tokens=out_len, k=4)
        spec_time = time.time() - spec_start
        spec_tokens = len(generated_ids)
        spec_speed = spec_tokens / spec_time if spec_time > 0 else 0.0
        
        speedup = spec_speed / std_speed if std_speed > 0 else 1.0
        print(f"  Speculative Speed: {spec_speed:.2f} tok/s | Latency: {spec_time * 1000.0:.1f} ms")
        print(f"  Acceptance Rate: {spec_metrics['acceptance_rate'] * 100.0:.1f}% | Speedup: {speedup:.2f}x")
        
        results.append({
            "output_len": out_len,
            "std_latency_ms": std_time * 1000.0,
            "std_speed_tok_sec": std_speed,
            "spec_latency_ms": spec_time * 1000.0,
            "spec_speed_tok_sec": spec_speed,
            "speedup": speedup,
            "acceptance_rate": spec_metrics["acceptance_rate"],
            "spec_steps": spec_metrics["spec_steps"],
            "accepted_tokens": spec_metrics["accepted_tokens"]
        })
        
    # Save results to reports
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/speculative_benchmark_results.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nAll speculative benchmarks finished! Saved report to {report_path}")

if __name__ == "__main__":
    main()
