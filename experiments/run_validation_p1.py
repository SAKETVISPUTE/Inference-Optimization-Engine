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
    # Qwen 2.5-0.5B is small enough to load quickly on CPU.
    print(f"--- Loading model {model_id} ---")
    model, tokenizer = load_model_and_tokenizer(model_id=model_id, device=device, dtype="float32")
    
    generator = CustomGenerator(model, tokenizer)
    
    prompt = "State the first three numbers in the sequence of prime numbers."
    max_new_tokens = 20
    
    print("\n--- Running Custom Greedy Generation ---")
    custom_res = generator.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,  # Greedy
        use_cache=False
    )
    print("Custom Greedy Output:", repr(custom_res["decoded_text"]))
    print("Custom Greedy Tokens:", custom_res["generated_tokens"])
    
    print("\n--- Running Hugging Face Default Generation ---")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    hf_outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # Greedy
        repetition_penalty=1.0,
        use_cache=False
    )
    # Exclude prompt tokens to align with new tokens generated
    prompt_len = inputs["input_ids"].shape[-1]
    hf_generated_tokens = hf_outputs[0][prompt_len:].tolist()
    hf_decoded_text = tokenizer.decode(hf_generated_tokens, skip_special_tokens=True)
    print("HF Greedy Output:", repr(hf_decoded_text))
    print("HF Greedy Tokens:", hf_generated_tokens)
    
    # Assert parity
    print("\n--- Comparing Outputs ---")
    match = (custom_res["generated_tokens"] == hf_generated_tokens)
    print("Tokens Match:", match)
    if match:
        print("Success: Custom autoregressive generation matches Hugging Face greedy decoding exactly!")
    else:
        print("Warning: Logits/token mismatch between Custom and HF!")
        # Print differences
        for idx, (c_tok, h_tok) in enumerate(zip(custom_res["generated_tokens"], hf_generated_tokens)):
            if c_tok != h_tok:
                print(f"Diff at index {idx}: Custom={c_tok} ('{tokenizer.decode([c_tok])}'), HF={h_tok} ('{tokenizer.decode([h_tok])}')")
                
    # Run Sampling Demo
    print("\n--- Running Custom Sampling (Temp=0.7, Top-P=0.9, Top-K=50) ---")
    sampling_res = generator.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        use_cache=False
    )
    print("Sampling Output:", repr(sampling_res["decoded_text"]))
    print("Sampling Tokens:", sampling_res["generated_tokens"])

if __name__ == "__main__":
    main()
