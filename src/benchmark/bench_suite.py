import time
import torch
import gc
from typing import Dict, Any, List

from src.models.loader import load_model_and_tokenizer
from src.decoder.generator import CustomGenerator
from src.quantization.int8 import quantize_model_to_int8
from src.quantization.int4 import quantize_model_to_int4
from src.profiling.gpu_profiler import reset_cuda_memory_stats, get_cuda_max_memory_allocated_mb

def run_benchmark_case(
    model, 
    tokenizer, 
    context_len: int, 
    output_len: int, 
    use_cache: bool
) -> Dict[str, float]:
    """
    Runs a single benchmark execution and measures metrics.
    """
    device = next(model.parameters()).device
    model.eval()
    
    # 1. Generate a dummy input sequence of exact length
    dummy_input_ids = torch.randint(low=10, high=1000, size=(1, context_len), dtype=torch.long, device=device)
    dummy_attention_mask = torch.ones_like(dummy_input_ids)
    
    # Garbage collect and reset CUDA memory stats
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        reset_cuda_memory_stats()
        
    start_vram = get_cuda_max_memory_allocated_mb()
    
    # Initialize generator variables
    curr_input_ids = dummy_input_ids.clone()
    curr_attention_mask = dummy_attention_mask.clone()
    past_key_values = None
    step_times = []
    
    # Run loop
    with torch.no_grad():
        for step in range(output_len):
            step_start = time.perf_counter()
            
            if use_cache:
                if step == 0:
                    outputs = model(
                        input_ids=curr_input_ids,
                        attention_mask=curr_attention_mask,
                        use_cache=True
                    )
                else:
                    next_input_ids = curr_input_ids[:, -1:]
                    outputs = model(
                        input_ids=next_input_ids,
                        attention_mask=curr_attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
            else:
                outputs = model(input_ids=curr_input_ids, attention_mask=curr_attention_mask)
                logits = outputs.logits[:, -1, :]
                
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            curr_input_ids = torch.cat([curr_input_ids, next_token], dim=-1)
            
            # Update attention mask
            ones = torch.ones((curr_attention_mask.shape[0], 1), dtype=curr_attention_mask.dtype, device=device)
            curr_attention_mask = torch.cat([curr_attention_mask, ones], dim=-1)
            
            step_end = time.perf_counter()
            step_times.append(step_end - step_start)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
    total_time_s = sum(step_times)
    ttft_ms = step_times[0] * 1000.0
    
    if len(step_times) > 1:
        tpot_ms = (sum(step_times[1:]) / (len(step_times) - 1)) * 1000.0
        tokens_per_sec = (len(step_times) - 1) / sum(step_times[1:])
    else:
        tpot_ms = 0.0
        tokens_per_sec = 0.0
        
    end_vram = get_cuda_max_memory_allocated_mb()
    peak_vram_gb = (end_vram - start_vram) / 1024.0
    if peak_vram_gb < 0:
        peak_vram_gb = 0.0
        
    return {
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "tokens_per_sec": tokens_per_sec,
        "peak_vram_gb": peak_vram_gb,
        "total_latency_ms": total_time_s * 1000.0,
        "generated_len": len(step_times)
    }
