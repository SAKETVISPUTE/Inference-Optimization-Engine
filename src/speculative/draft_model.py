import torch
from typing import List, Tuple, Any

class DraftModelWrapper:
    def __init__(self, model, tokenizer, device: str = "cuda:0"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def generate_candidates(
        self, 
        input_ids: torch.Tensor, 
        k: int
    ) -> Tuple[List[int], Any]:
        """
        Generates k draft candidate tokens autoregressively.
        Returns:
            List of generated token IDs (length k)
            Updated draft past_key_values
        """
        draft_tokens = []
        curr_input_ids = input_ids.clone()
        past_key_values = None
        
        # Use KV cache for the draft model to make it as fast as possible
        with torch.no_grad():
            for step in range(k):
                if step == 0:
                    # Prefill pass for the new inputs
                    outputs = self.model(input_ids=curr_input_ids, use_cache=True)
                else:
                    # Single-token decode pass
                    outputs = self.model(
                        input_ids=curr_input_ids[:, -1:], 
                        past_key_values=past_key_values, 
                        use_cache=True
                    )
                    
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                draft_tokens.append(next_token.item())
                curr_input_ids = torch.cat([curr_input_ids, next_token], dim=-1)
                
        return draft_tokens, past_key_values
