import time
import torch
from typing import List, Dict, Any
from src.scheduler.request import Request
from src.scheduler.request_queue import RequestQueue
from src.scheduler.batch_manager import BatchManager

class Scheduler:
    def __init__(self, model, tokenizer, device: str = "cuda:0"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def run_sequential(self, requests: List[Request]) -> List[Request]:
        """
        Processes requests one by one sequentially.
        """
        print(f"Running {len(requests)} requests sequentially...")
        for req in requests:
            req.start()
            
            # Prefill step
            prompt_tensor = torch.tensor([req.prompt_ids], dtype=torch.long, device=self.device)
            with torch.no_grad():
                outputs = self.model(input_ids=prompt_tensor, use_cache=True)
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
            req.add_token(next_token.item())
            
            # Decode loop
            while len(req.generated_ids) < req.max_new_tokens:
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=next_token,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    past_key_values = outputs.past_key_values
                    logits = outputs.logits[:, -1, :]
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                req.add_token(next_token.item())
                
            req.finish()
        return requests

    def run_static_batch(self, requests: List[Request]) -> List[Request]:
        """
        Processes all requests together using left-padding static batching.
        """
        if not requests:
            return requests
            
        print(f"Running {len(requests)} requests via static batching...")
        for req in requests:
            req.start()

        # Find max prompt length for padding
        max_prompt_len = max(len(req.prompt_ids) for req in requests)
        
        # Pad prompts on the left
        padded_prompts = []
        attention_masks = []
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        
        for req in requests:
            padding_len = max_prompt_len - len(req.prompt_ids)
            padded_prompt = [pad_token_id] * padding_len + req.prompt_ids
            attention_mask = [0] * padding_len + [1] * len(req.prompt_ids)
            padded_prompts.append(padded_prompt)
            attention_masks.append(attention_mask)
            
        input_ids = torch.tensor(padded_prompts, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=self.device)
        
        # Prefill step
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            next_tokens = torch.argmax(logits, dim=-1, keepdim=True)
            
        # Update requests
        for i, req in enumerate(requests):
            req.add_token(next_tokens[i].item())

        # Prepare for decoding
        active_mask = [True] * len(requests)
        curr_tokens = next_tokens
        
        # Decode loop
        max_new_tokens = max(req.max_new_tokens for req in requests)
        for step in range(1, max_new_tokens):
            if not any(active_mask):
                break
                
            # Update attention mask for next step
            ones = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=self.device)
            attention_mask = torch.cat([attention_mask, ones], dim=-1)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=curr_tokens,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
                next_tokens = torch.argmax(logits, dim=-1, keepdim=True)
                
            curr_tokens = next_tokens
            
            # Record generated tokens
            for i, req in enumerate(requests):
                if active_mask[i]:
                    token_id = next_tokens[i].item()
                    req.add_token(token_id)
                    
                    if token_id == self.tokenizer.eos_token_id or len(req.generated_ids) >= req.max_new_tokens:
                        active_mask[i] = False
                        req.finish()
                        
        # Ensure all requests are finished
        for req in requests:
            if req.status != "FINISHED":
                req.finish()
                
        return requests

    def run_continuous_batch(self, requests: List[Request], max_batch_size: int = 4) -> List[Request]:
        """
        Processes requests using continuous (iteration-level) batching.
        New requests are dynamically scheduled as active requests finish.
        """
        print(f"Running {len(requests)} requests via continuous batching...")
        queue = RequestQueue()
        for req in requests:
            queue.add_request(req)
            
        batch_manager = BatchManager(max_batch_size=max_batch_size)
        
        # We store past_key_values and next_tokens for each request in dictionaries
        pkvs: Dict[str, Any] = {}
        curr_tokens: Dict[str, torch.Tensor] = {}
        
        while queue.has_pending() or len(batch_manager.get_active_requests()) > 0:
            # 1. Fill empty slots in the batch manager
            free_slots = batch_manager.get_free_slots()
            if free_slots > 0 and queue.has_pending():
                new_reqs = queue.pop_pending(limit=free_slots)
                for req in new_reqs:
                    batch_manager.add_request(req)
                    # Run prefill step for this new request
                    prompt_tensor = torch.tensor([req.prompt_ids], dtype=torch.long, device=self.device)
                    with torch.no_grad():
                        outputs = self.model(input_ids=prompt_tensor, use_cache=True)
                        pkvs[req.request_id] = outputs.past_key_values
                        logits = outputs.logits[:, -1, :]
                        next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    req.add_token(next_token.item())
                    curr_tokens[req.request_id] = next_token

            active_requests = batch_manager.get_active_requests()
            if not active_requests:
                continue

            # 2. Run 1 step of decoding for all active requests
            # To simulate continuous batching on the GPU, we run the model for each active request.
            # (Note: In production serving systems, this decoding step is batch-executed in a single forward pass
            # using PagedAttention. We perform the logical iteration-level scheduling here).
            for req in active_requests:
                next_token = curr_tokens[req.request_id]
                past_key_values = pkvs[req.request_id]
                
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=next_token,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    pkvs[req.request_id] = outputs.past_key_values
                    logits = outputs.logits[:, -1, :]
                    next_token_out = torch.argmax(logits, dim=-1, keepdim=True)
                    
                req.add_token(next_token_out.item())
                curr_tokens[req.request_id] = next_token_out
                
                # Check if finished
                if next_token_out.item() == self.tokenizer.eos_token_id or len(req.generated_ids) >= req.max_new_tokens:
                    req.finish()
                    # Clean up cache reference
                    pkvs.pop(req.request_id, None)
                    curr_tokens.pop(req.request_id, None)
                    
            # 3. Evict finished requests
            batch_manager.remove_finished()
            
        return requests
