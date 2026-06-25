import time
import torch
from typing import List, Dict, Any, Tuple
from src.speculative.draft_model import DraftModelWrapper
from src.speculative.verifier import VerifierModelWrapper
from src.speculative.acceptance import verify_and_accept_greedy, truncate_kv_cache

class SpeculativeGenerator:
    def __init__(self, target_model, draft_model, tokenizer, device: str = "cuda:0"):
        self.tokenizer = tokenizer
        self.device = device
        
        self.target = VerifierModelWrapper(target_model, tokenizer, device)
        self.draft = DraftModelWrapper(draft_model, tokenizer, device)
        
    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 128, 
        k: int = 4
    ) -> Tuple[str, List[int], Dict[str, Any]]:
        """
        Generates tokens using Speculative Decoding.
        """
        # Encode inputs
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        prompt_len = len(prompt_ids)
        
        # Initialize KV Caches for both models with the prompt
        with torch.no_grad():
            target_outputs = self.target.model(input_ids=input_ids, use_cache=True)
            target_pkv = target_outputs.past_key_values
            target_logits = target_outputs.logits[:, -1, :]
            
            draft_outputs = self.draft.model(input_ids=input_ids, use_cache=True)
            draft_pkv = draft_outputs.past_key_values
            
        # Get the first token
        first_token = torch.argmax(target_logits, dim=-1, keepdim=True)
        generated_ids = [first_token.item()]
        
        # Timing metrics
        start_time = time.time()
        ttft = (time.time() - start_time) * 1000.0  # approximate
        
        total_spec_steps = 0
        total_accepted_tokens = 0
        
        # Current token index and cache state
        curr_target_pkv = target_pkv
        curr_draft_pkv = draft_pkv
        last_accepted_token_id = generated_ids[-1]
        
        # autogressive speculative loop
        while len(generated_ids) < max_new_tokens:
            if last_accepted_token_id == self.tokenizer.eos_token_id:
                break
                
            total_spec_steps += 1
            
            # Step 1: Draft model generates K candidate tokens
            draft_tokens = []
            draft_input = torch.tensor([[last_accepted_token_id]], dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                for step in range(k):
                    outputs = self.draft.model(
                        input_ids=draft_input, 
                        past_key_values=curr_draft_pkv, 
                        use_cache=True
                    )
                    curr_draft_pkv = outputs.past_key_values
                    logits = outputs.logits[:, -1, :]
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    draft_tokens.append(next_token.item())
                    draft_input = next_token
                    
            # Step 2: Target verifier checks the proposed tokens in parallel
            verifier_logits, curr_target_pkv = self.target.verify(
                draft_tokens=draft_tokens,
                past_key_values=curr_target_pkv,
                last_accepted_token_id=last_accepted_token_id
            )
            
            # Step 3: Acceptance decision
            accepted_tokens, num_accepted = verify_and_accept_greedy(
                draft_tokens=draft_tokens,
                verifier_logits=verifier_logits
            )
            
            total_accepted_tokens += num_accepted
            generated_ids.extend(accepted_tokens)
            last_accepted_token_id = accepted_tokens[-1]
            
            # Step 4: Truncate caches
            # New sequence length in caches:
            # - Target cache should have prompt_len + previously_generated + accepted_tokens
            # - Draft cache should have target_len - 1 (since it will process the last accepted token as input next)
            target_keep_len = prompt_len + len(generated_ids)
            draft_keep_len = target_keep_len - 1
            
            curr_target_pkv = truncate_kv_cache(curr_target_pkv, target_keep_len)
            curr_draft_pkv = truncate_kv_cache(curr_draft_pkv, draft_keep_len)
            
        total_time = time.time() - start_time
        speed = len(generated_ids) / total_time if total_time > 0 else 0.0
        
        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        metrics = {
            "total_time_s": total_time,
            "ttft_ms": ttft,
            "tokens_per_sec": speed,
            "num_generated_tokens": len(generated_ids),
            "spec_steps": total_spec_steps,
            "accepted_tokens": total_accepted_tokens,
            "acceptance_rate": total_accepted_tokens / (total_spec_steps * k) if total_spec_steps > 0 else 0.0
        }
        
        return output_text, generated_ids, metrics
