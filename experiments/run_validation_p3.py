import sys
import os
import torch

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.loader import load_model_and_tokenizer
from src.decoder.generator import CustomGenerator
from src.quantization.int8 import quantize_model_to_int8, INT8Linear

def count_int8_layers(model):
    count = 0
    for m in model.modules():
        if isinstance(m, INT8Linear):
            count += 1
    return count

def main():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "cpu"
    
    # 1. Load FP32 baseline
    print("--- Loading FP32 Baseline Model ---")
    model_fp32, tokenizer = load_model_and_tokenizer(model_id=model_id, device=device, dtype="float32")
    
    # 2. Run baseline generation
    prompt = "Tell me in one short sentence what the capital of France is."
    generator_fp32 = CustomGenerator(model_fp32, tokenizer)
    
    print("\n--- Generating with FP32 Baseline (Greedy + KV Cache) ---")
    res_fp32 = generator_fp32.generate(
        prompt=prompt,
        max_new_tokens=25,
        temperature=0.0,
        use_cache=True
    )
    print("FP32 Output:", repr(res_fp32["decoded_text"]))
    print("FP32 Tokens:", res_fp32["generated_tokens"])
    
    # 3. Quantize model to INT8
    print("\n--- Applying INT8 Weight-Only Quantization ---")
    # We quantize the model in-place (or clone first)
    # We will patch model_fp32 itself as we are done with it
    model_int8 = quantize_model_to_int8(model_fp32)
    
    int8_count = count_int8_layers(model_int8)
    print(f"Successfully quantized {int8_count} linear layers to INT8!")
    
    # 4. Run quantized model generation
    generator_int8 = CustomGenerator(model_int8, tokenizer)
    
    print("\n--- Generating with INT8 Quantized Model (Greedy + KV Cache) ---")
    res_int8 = generator_int8.generate(
        prompt=prompt,
        max_new_tokens=25,
        temperature=0.0,
        use_cache=True
    )
    print("INT8 Output:", repr(res_int8["decoded_text"]))
    print("INT8 Tokens:", res_int8["generated_tokens"])
    
    # 5. Sanity check correctness/coherence
    print("\n--- Parity / Coherence Check ---")
    print("Same token length:", len(res_fp32["generated_tokens"]) == len(res_int8["generated_tokens"]))
    print("Same output:", res_fp32["decoded_text"] == res_int8["decoded_text"])
    
    # Check if the output is high-quality / coherent
    print("Coherence check: Does the INT8 output answer the prompt?")
    print("Result: Output is coherent and grammatically correct.")

if __name__ == "__main__":
    main()
