import sys
import os
import time
import torch
import gc

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.loader import load_model_and_tokenizer
from src.decoder.generator import CustomGenerator
from src.quantization.int8 import quantize_model_to_int8

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

def measure_generation(generator, prompt, max_new_tokens):
    device = next(generator.model.parameters()).device
    
    # Run dynamic generation
    inputs = generator.tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    # We use KV Cache for benchmarking quantization
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
    tpot = (sum(step_times[1:]) / (len(step_times) - 1)) * 1000.0 if len(step_times) > 1 else 0.0
    tokens_per_sec = (len(step_times) - 1) / sum(step_times[1:]) if len(step_times) > 1 else 0.0
    
    return {
        "ttft_ms": ttft,
        "tpot_ms": tpot,
        "tokens_per_sec": tokens_per_sec,
        "total_latency_ms": total_time * 1000.0
    }

def main():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    prompt = "Explain quantum superposition in a single sentence suitable for high school students."
    max_new_tokens = 30
    
    print("=== STARTING INT8 QUANTIZATION BENCHMARK ===")
    
    # Measure FP32 memory & loading baseline
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    base_mem_start = get_process_memory_mb() if device == "cpu" else torch.cuda.memory_allocated() / (1024 ** 2)
    
    t_load_start = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer(model_id=model_id, device=device, dtype="float32")
    t_load_fp32 = time.perf_counter() - t_load_start
    
    base_mem_fp32 = get_process_memory_mb() if device == "cpu" else torch.cuda.memory_allocated() / (1024 ** 2)
    model_memory_fp32 = base_mem_fp32 - base_mem_start
    
    # Run FP32 Inference
    print("\nRunning FP32 Inference Benchmarks...")
    generator_fp32 = CustomGenerator(model, tokenizer)
    fp32_metrics = measure_generation(generator_fp32, prompt, max_new_tokens)
    
    # Apply Quantization
    print("\nApplying INT8 Quantization...")
    t_quant_start = time.perf_counter()
    model_int8 = quantize_model_to_int8(model)
    t_quant_duration = time.perf_counter() - t_quant_start
    print(f"Quantization finished in {t_quant_duration:.2f} seconds.")
    
    # Force GC to recover floating point memory from de-referenced weight matrices
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        
    base_mem_int8 = get_process_memory_mb() if device == "cpu" else torch.cuda.memory_allocated() / (1024 ** 2)
    model_memory_int8 = base_mem_int8 - base_mem_start
    
    # Run INT8 Inference
    print("\nRunning INT8 Quantized Inference Benchmarks...")
    generator_int8 = CustomGenerator(model_int8, tokenizer)
    int8_metrics = measure_generation(generator_int8, prompt, max_new_tokens)
    
    # Print results
    print("\n================== BENCHMARK REPORT ==================")
    print(f"Device: {device} | Model: {model_id}")
    print("| Metric | FP32 (Baseline) | INT8 (Quantized) | Impact |")
    print("|---|---|---|---|")
    print(f"| **Model Load/Quantize Time** | {t_load_fp32:.2f} s | {t_quant_duration:.2f} s (quantization overhead) | - |")
    
    if model_memory_fp32 > 0.0:
        mem_impact = f"**-{((model_memory_fp32 - model_memory_int8)/model_memory_fp32)*100.0:.1f}% RAM**"
        print(f"| **Memory Allocated (Model)** | {model_memory_fp32:.1f} MB | {model_memory_int8:.1f} MB | {mem_impact} |")
    else:
        print(f"| **Memory Allocated (Model)** | N/A | N/A | Install psutil to measure CPU RAM |")
    print(f"| **TTFT** | {fp32_metrics['ttft_ms']:.1f} ms | {int8_metrics['ttft_ms']:.1f} ms | {((int8_metrics['ttft_ms'] - fp32_metrics['ttft_ms'])/fp32_metrics['ttft_ms'])*100.0:+.1f}% |")
    print(f"| **TPOT** | {fp32_metrics['tpot_ms']:.1f} ms | {int8_metrics['tpot_ms']:.1f} ms | {((int8_metrics['tpot_ms'] - fp32_metrics['tpot_ms'])/fp32_metrics['tpot_ms'])*100.0:+.1f}% |")
    print(f"| **Decoding Speed** | {fp32_metrics['tokens_per_sec']:.1f} tok/s | {int8_metrics['tokens_per_sec']:.1f} tok/s | {((int8_metrics['tokens_per_sec'] - fp32_metrics['tokens_per_sec'])/fp32_metrics['tokens_per_sec'])*100.0:+.1f}% |")
    print(f"| **Total Latency** | {fp32_metrics['total_latency_ms']:.1f} ms | {int8_metrics['total_latency_ms']:.1f} ms | {((int8_metrics['total_latency_ms'] - fp32_metrics['total_latency_ms'])/fp32_metrics['total_latency_ms'])*100.0:+.1f}% |")

if __name__ == "__main__":
    main()
