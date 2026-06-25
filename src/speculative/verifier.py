import torch
from typing import Tuple, Any

class VerifierModelWrapper:
    def __init__(self, model, tokenizer, device: str = "cuda:0"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def verify(
        self,
        draft_tokens: list,
        past_key_values: Any,
        last_accepted_token_id: int
    ) -> Tuple[torch.Tensor, Any]:
        """
        Runs the target verifier model in parallel on the draft tokens.
        Args:
            draft_tokens: list of k token IDs proposed by the draft model.
            past_key_values: target model's KV cache.
            last_accepted_token_id: the last token ID that was verified/accepted.
        Returns:
            verifier_logits: shape (1, k + 1, vocab_size) representing target model predictions.
            updated_past_key_values: target model's KV cache after appending draft tokens.
        """
        # We concatenate the last accepted token and the proposed draft tokens
        input_list = [last_accepted_token_id] + draft_tokens
        input_ids = torch.tensor([input_list], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            # If past_key_values is None, we are running prefill on the prompt.
            # In speculative decoding, prefill is done once, and then verification is run.
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            
        return outputs.logits, outputs.past_key_values
