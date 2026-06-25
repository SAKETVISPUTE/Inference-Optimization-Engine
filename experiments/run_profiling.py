import os
import sys
import torch
import gc

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.loader import load_model_and_tokenizer
from src.decoder.generator import CustomGenerator
from src.quantization.int8 import quantize_model_to_int8
from src.quantization.int4 import quantize_model_to_int4
from src.profiling.gpu_profiler import GPUProfiler

def main():
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. GPU is required for profiling.")
        sys.exit(1)
        
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    device = "cuda:0"
    
    print(f"=== Starting GPU Profiling Suite for {model_id} ===")
    
    prompt = "Explain quantum superposition in a single sentence."
    max_new_tokens = 16  # Profiling only requires a short sequence to inspect trace signatures
    
    configs = {
        "fp32_nocache": {"use_cache": False, "quantize": None},
        "fp32_cache": {"use_cache": True, "quantize": None},
        "int8_cache": {"use_cache": True, "quantize": "int8"},
        "int4_cache": {"use_cache": True, "quantize": "int4"},
    }
    
    for name, cfg in configs.items():
        print(f"\n--- Profiling configuration: {name} ---")
        
        # Load model
        print("Loading model...")
        model, tokenizer = load_model_and_tokenizer(model_id=model_id, device=device, dtype="float32")
        
        # Apply quantization
        if cfg["quantize"] == "int8":
            print("Applying INT8 Quantization...")
            model = quantize_model_to_int8(model)
        elif cfg["quantize"] == "int4":
            print("Applying INT4 Quantization...")
            model = quantize_model_to_int4(model)
            
        generator = CustomGenerator(model, tokenizer)
        
        # Warm-up to trigger lazy loading / CUDA context initialization before profiling starts
        print("Warming up...")
        _ = generator.generate(prompt=prompt, max_new_tokens=4, use_cache=cfg["use_cache"])
        
        # Profile generation
        print(f"Starting torch.profiler trace for {name}...")
        with GPUProfiler(name=name, output_dir="reports/traces"):
            _ = generator.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                use_cache=cfg["use_cache"]
            )
            
        # Clean up memory
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
    print("\nProfiling completed! Traces saved to reports/traces/")

if __name__ == "__main__":
    main()
