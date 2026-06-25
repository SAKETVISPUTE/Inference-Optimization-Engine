import sys
import os
import time
import torch
import gc

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.loader import load_model_and_tokenizer
from src.decoder.generator import CustomGenerator

# Try importing psutil for RAM tracking
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

def get_process_memory_mb():
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 2)
    return 0.0

def measure_generation(generator, prompt, max_new_tokens, use_cache):
    device = next(generator.model.parameters()).device
    
    # Garbage collect and reset CUDA memory stats
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
    start_mem = get_process_memory_mb() if device.type == "cpu" else torch.cuda.memory_allocated() / (1024 ** 2)
    
    # Tokenize prompt to get length
    inputs = generator.tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    prompt_len = input_ids.shape[-1]
    
    # Warmup / Prefill
    t0 = time.perf_counter()
    
    # Run custom loop manually step-by-step to isolate TTFT and subsequent token latencies
    generator.model.eval()
    curr_input_ids = input_ids.clone()
    curr_attention_mask = inputs.get("attention_mask", None)
    if curr_attention_mask is not None:
        curr_attention_mask = curr_attention_mask.to(device)
        
    past_key_values = None
    step_times = []
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            step_start = time.perf_counter()
            
            if use_cache:
                if step == 0:
                    outputs = generator.model(
                        input_ids=curr_input_ids,
                        attention_mask=curr_attention_mask,
                        use_cache=True
                    )
                else:
                    next_input_ids = curr_input_ids[:, -1:]
                    outputs = generator.model(
                        input_ids=next_input_ids,
                        attention_mask=curr_attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
            else:
                outputs = generator.model(input_ids=curr_input_ids, attention_mask=curr_attention_mask)
                logits = outputs.logits[:, -1, :]
                
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            curr_input_ids = torch.cat([curr_input_ids, next_token], dim=-1)
            
            if curr_attention_mask is not None:
                ones = torch.ones((curr_attention_mask.shape[0], 1), dtype=curr_attention_mask.dtype, device=device)
                curr_attention_mask = torch.cat([curr_attention_mask, ones], dim=-1)
                
            step_end = time.perf_counter()
            step_times.append(step_end - step_start)
            
            if next_token.item() == generator.tokenizer.eos_token_id:
                break
                
    total_time = sum(step_times)
    ttft = step_times[0] * 1000.0  # ms
    
    # Calculate TPOT (Time Per Output Token) on remaining tokens
    if len(step_times) > 1:
        tpot = (sum(step_times[1:]) / (len(step_times) - 1)) * 1000.0  # ms
        tokens_per_sec = (len(step_times) - 1) / sum(step_times[1:])
    else:
        tpot = 0.0
        tokens_per_sec = 0.0
        
    end_mem = get_process_memory_mb() if device.type == "cpu" else torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_mem_increase = max(0.0, end_mem - start_mem)
    
    return {
        "ttft_ms": ttft,
        "tpot_ms": tpot,
        "tokens_per_sec": tokens_per_sec,
        "total_latency_ms": total_time * 1000.0,
        "peak_mem_increase_mb": peak_mem_increase,
        "generated_length": len(step_times)
    }

def main():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"--- Loading model {model_id} on {device} ---")
    model, tokenizer = load_model_and_tokenizer(model_id=model_id, device=device, dtype="float32")
    generator = CustomGenerator(model, tokenizer)
    
    # Define experimental settings
    # We will use prompt sizes of different lengths to show the quadratic vs linear growth
    prompts = {
        "Short (16 tokens)": "Explain gravity.",
        "Medium (128 tokens)": "Explain gravity. " * 8,
        "Long (256 tokens)": "Explain gravity. " * 16,
    }
    
    max_new_tokens = 30
    
    results = {}
    
    for name, text in prompts.items():
        print(f"\nBenchmarking prompt: {name} (Approx tokens: {len(tokenizer.encode(text))})")
        results[name] = {}
        
        # Benchmark without KV Cache
        print("  Running WITHOUT KV Cache...")
        no_cache_res = measure_generation(generator, text, max_new_tokens, use_cache=False)
        results[name]["no_cache"] = no_cache_res
        
        # Benchmark with KV Cache
        print("  Running WITH KV Cache...")
        with_cache_res = measure_generation(generator, text, max_new_tokens, use_cache=True)
        results[name]["with_cache"] = with_cache_res
        
    # Output markdown report
    print("\n================== BENCHMARK REPORT ==================")
    print(f"Device: {device} | Model: {model_id}")
    print("| Prompt Length | Cache Mode | TTFT (ms) | TPOT (ms) | Speed (tok/s) | Peak Mem Delta | Speedup |")
    print("|---|---|---|---|---|---|---|")
    
    for name in prompts.keys():
        nc = results[name]["no_cache"]
        wc = results[name]["with_cache"]
        
        speedup = nc["tpot_ms"] / wc["tpot_ms"] if wc["tpot_ms"] > 0 else 1.0
        
        # Format rows
        print(f"| {name} | No Cache | {nc['ttft_ms']:.1f} | {nc['tpot_ms']:.1f} | {nc['tokens_per_sec']:.1f} | {nc['peak_mem_increase_mb']:.2f} MB | baseline |")
        print(f"| {name} | KV Cache | {wc['ttft_ms']:.1f} | {wc['tpot_ms']:.1f} | {wc['tokens_per_sec']:.1f} | {wc['peak_mem_increase_mb']:.2f} MB | **{speedup:.2f}x** |")
        print("| | | | | | | |")

if __name__ == "__main__":
    main()
