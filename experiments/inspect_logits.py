import sys
import os
import torch

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.loader import load_model_and_tokenizer

def main():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "cpu"
    model, tokenizer = load_model_and_tokenizer(model_id=model_id, device=device, dtype="float32")
    
    prompt = "State the first three numbers in the sequence of prime numbers."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    print("\nModel Generation Config:")
    print(model.generation_config)
    
    # Let's run a 16-step generation step-by-step using both methods and compare the outputs/logits.
    # We will run Hugging Face generate with return_dict_in_generate=True, output_scores=True to get logits.
    hf_outputs = model.generate(
        **inputs,
        max_new_tokens=16,
        do_sample=False,
        use_cache=False,
        repetition_penalty=1.0,
        return_dict_in_generate=True,
        output_scores=True
    )
    
    # Custom step-by-step
    curr_input_ids = input_ids.clone()
    curr_attention_mask = attention_mask.clone()
    
    print("\n--- Comparing logits step by step ---")
    for step in range(16):
        # Our custom forward pass
        with torch.no_grad():
            outputs = model(input_ids=curr_input_ids, attention_mask=curr_attention_mask)
            custom_logits = outputs.logits[:, -1, :]
            
        # HF logits for this step
        hf_logits = hf_outputs.scores[step]
        
        # Check difference
        abs_diff = torch.abs(custom_logits - hf_logits)
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        
        # Get predictions
        custom_pred = torch.argmax(custom_logits, dim=-1).item()
        hf_pred = torch.argmax(hf_logits, dim=-1).item()
        
        custom_token_str = tokenizer.decode([custom_pred])
        hf_token_str = tokenizer.decode([hf_pred])
        
        print(f"Step {step:02d}: Max Diff = {max_diff:.6e}, Mean Diff = {mean_diff:.6e} | Custom Token = {custom_pred} ({repr(custom_token_str)}), HF Token = {hf_pred} ({repr(hf_token_str)})")
        
        # Update input ids for custom loop using custom pred to follow its own path
        curr_input_ids = torch.cat([curr_input_ids, torch.tensor([[custom_pred]], device=device)], dim=-1)
        curr_attention_mask = torch.cat([curr_attention_mask, torch.ones((1, 1), dtype=curr_attention_mask.dtype, device=device)], dim=-1)

if __name__ == "__main__":
    main()
