import os
import sys
import json
import torch
import gc

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.loader import load_model_and_tokenizer
from src.benchmark.bench_suite import run_benchmark_case
from src.quantization.int8 import quantize_model_to_int8
from src.quantization.int4 import quantize_model_to_int4

def main():
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. GPU is required for this benchmarking script.")
        sys.exit(1)
        
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    device = "cuda:0"
    
    print(f"=== Starting GPU Benchmark Suite for {model_id} ===")
    
    # 1. Sweep configurations
    context_lens = [512, 2048, 4096, 8192, 16384]
    output_lens = [32, 128, 512, 1024]
    
    configs = ["FP32 (No Cache)", "FP32 (KV Cache)", "INT8 (KV Cache)", "INT4 (KV Cache)"]
    
    results = []
    
    # Track baseline GPU memory usage before model load
    gc.collect()
    torch.cuda.empty_cache()
    start_vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    
    # Loop over configs
    for config_name in configs:
        print(f"\n--- Setting up Configuration: {config_name} ---")
        
        # Load fresh model to prevent memory leaks/overlaps across quantizations
        # Qwen 2.5-3B is small enough to download and load quickly on high-speed server links
        print("Loading base model...")
        model, tokenizer = load_model_and_tokenizer(model_id=model_id, device=device, dtype="float32")
        
        # Apply quantization if specified
        if "INT8" in config_name:
            print("Applying INT8 Quantization...")
            model = quantize_model_to_int8(model)
        elif "INT4" in config_name:
            print("Applying INT4 Quantization...")
            model = quantize_model_to_int4(model)
            
        use_cache = "KV Cache" in config_name
        
        # Sweep variables
        for ctx in context_lens:
            # Skip massive context sizes for No-Cache to avoid huge execution times/OOMs
            if not use_cache and ctx > 2048:
                continue
                
            for out in output_lens:
                # Skip massive output sizes for No-Cache to avoid huge execution times
                if not use_cache and out > 128:
                    continue
                    
                print(f"Benchmarking: Context={ctx} | Output={out} | Cache={use_cache}...")
                
                try:
                    stats = run_benchmark_case(
                        model=model,
                        tokenizer=tokenizer,
                        context_len=ctx,
                        output_len=out,
                        use_cache=use_cache
                    )
                    
                    record = {
                        "config": config_name,
                        "context_len": ctx,
                        "output_len": out,
                        "use_cache": use_cache,
                        "ttft_ms": stats["ttft_ms"],
                        "tpot_ms": stats["tpot_ms"],
                        "tokens_per_sec": stats["tokens_per_sec"],
                        "peak_vram_gb": stats["peak_vram_gb"],
                        "total_latency_ms": stats["total_latency_ms"]
                    }
                    
                    results.append(record)
                    print(f"  Result -> Speed: {stats['tokens_per_sec']:.2f} tok/s | Peak VRAM: {stats['peak_vram_gb']:.2f} GB | Latency: {stats['total_latency_ms']:.1f} ms")
                    
                except Exception as e:
                    print(f"  Failed: {str(e)}")
                    results.append({
                        "config": config_name,
                        "context_len": ctx,
                        "output_len": out,
                        "use_cache": use_cache,
                        "error": str(e)
                    })
                    
        # Cleanup model from GPU VRAM
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
    # Save results to reports
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/gpu_benchmark_results.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nAll benchmarks finished! Saved report to {report_path}")
    
    # Print Markdown summary table for context=2048, output=128
    print("\n================== BENCHMARK SUMMARY (Ctx=2048, Out=128) ==================")
    print("| Configuration | TTFT (ms) | TPOT (ms) | Speed (tok/s) | Peak VRAM Delta (GB) | Latency (ms) |")
    print("|---|---|---|---|---|---|")
    for r in results:
        if r.get("context_len") == 2048 and r.get("output_len") == 128 and "error" not in r:
            print(f"| {r['config']} | {r['ttft_ms']:.1f} | {r['tpot_ms']:.1f} | {r['tokens_per_sec']:.1f} | {r['peak_vram_gb']:.2f} | {r['total_latency_ms']:.1f} |")

if __name__ == "__main__":
    main()
