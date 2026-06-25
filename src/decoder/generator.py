import torch
import torch.nn.functional as F

class CustomGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        use_cache: bool = False,
    ):
        """
        Generates text autoregressively.
        
        Args:
            prompt (str): Input prompt string.
            max_new_tokens (int): Max number of new tokens to generate.
            temperature (float): Randomness scale (0.0 means greedy decoding).
            top_k (int): Keep only top k tokens (0 means disabled).
            top_p (float): Keep top tokens with cumulative probability >= top_p (0.0 means disabled).
            use_cache (bool): Whether to use Key-Value caching (implemented in Phase 2).
            
        Returns:
            dict: Generated text and metrics.
        """
        # Set model to evaluation mode
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # Tokenize the input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            
        prompt_len = input_ids.shape[-1]
        
        # Keep track of generated tokens
        generated_tokens = []
        
        curr_input_ids = input_ids.clone()
        curr_attention_mask = attention_mask.clone() if attention_mask is not None else None
        
        past_key_values = None
        
        for i in range(max_new_tokens):
            if use_cache:
                if i == 0:
                    # Prefill step: process full prompt and initialize cache
                    outputs = self.model(
                        input_ids=curr_input_ids,
                        attention_mask=curr_attention_mask,
                        use_cache=True
                    )
                else:
                    # Decoding step: process only the last generated token with past KV cache
                    # Get the last token
                    next_input_ids = curr_input_ids[:, -1:]
                    
                    # We pass the accumulated attention mask which covers the past + new token
                    outputs = self.model(
                        input_ids=next_input_ids,
                        attention_mask=curr_attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
            else:
                # No KV Cache: recalculate representations for the entire sequence from scratch
                outputs = self.model(input_ids=curr_input_ids, attention_mask=curr_attention_mask)
                logits = outputs.logits[:, -1, :]
            
            # Sample next token
            next_token = self.sample_next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)
            
            # Append next token to track
            generated_tokens.append(next_token.item())
            curr_input_ids = torch.cat([curr_input_ids, next_token], dim=-1)
            
            # Update attention mask to include the new position
            if curr_attention_mask is not None:
                ones = torch.ones((curr_attention_mask.shape[0], 1), dtype=curr_attention_mask.dtype, device=device)
                curr_attention_mask = torch.cat([curr_attention_mask, ones], dim=-1)
                
            # Check for EOS token
            if next_token.item() == self.tokenizer.eos_token_id:
                break
                
        decoded_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        full_text = self.tokenizer.decode(curr_input_ids[0], skip_special_tokens=True)
        
        return {
            "generated_tokens": generated_tokens,
            "decoded_text": decoded_text,
            "full_text": full_text,
            "prompt_length": prompt_len,
            "generated_length": len(generated_tokens)
        }

    def sample_next_token(self, logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0) -> torch.Tensor:
        """
        Samples the next token from logits.
        """
        # Greedy decoding
        if temperature == 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)
            
        # Apply temperature scaling
        logits = logits / temperature
        
        # Apply Top-K filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
            
        # Apply Top-P (nucleus) filtering
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep the first token that exceeds the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Scatter mask back to original logit orientation
            indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
            
        # Sample from the filtered distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token
