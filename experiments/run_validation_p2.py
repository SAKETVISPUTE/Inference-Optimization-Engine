import sys
import os
import torch

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.loader import load_model_and_tokenizer
from src.decoder.generator import CustomGenerator

def main():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "cpu"
    print(f"--- Loading model {model_id} for KV Cache validation ---")
    model, tokenizer = load_model_and_tokenizer(model_id=model_id, device=device, dtype="float32")
    
    generator = CustomGenerator(model, tokenizer)
    
    prompt = "Explain in one sentence what quantum computing is."
    max_new_tokens = 30
    
    print("\n--- Running Greedy WITHOUT KV Cache ---")
    res_no_cache = generator.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        use_cache=False
    )
    print("No-Cache Output:", repr(res_no_cache["decoded_text"]))
    print("No-Cache Tokens:", res_no_cache["generated_tokens"])
    
    print("\n--- Running Greedy WITH KV Cache ---")
    res_with_cache = generator.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        use_cache=True
    )
    print("Cache Output:   ", repr(res_with_cache["decoded_text"]))
    print("Cache Tokens:   ", res_with_cache["generated_tokens"])
    
    # Assert correctness
    print("\n--- Correctness Check ---")
    match = (res_no_cache["generated_tokens"] == res_with_cache["generated_tokens"])
    print("Tokens Match exactly:", match)
    if match:
        print("Success: KV Cache generation is 100% mathematically equivalent to baseline generation!")
    else:
        print("Error: Mismatch detected between cached and non-cached outputs!")
        # Print differences
        for idx, (c_tok, nc_tok) in enumerate(zip(res_with_cache["generated_tokens"], res_no_cache["generated_tokens"])):
            if c_tok != nc_tok:
                print(f"Diff at index {idx}: Cache={c_tok} ('{tokenizer.decode([c_tok])}'), No-Cache={nc_tok} ('{tokenizer.decode([nc_tok])}')")

if __name__ == "__main__":
    main()
