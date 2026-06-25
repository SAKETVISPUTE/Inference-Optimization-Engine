import torch
from typing import List, Tuple, Any

def truncate_kv_cache(past_key_values: Any, num_tokens_to_keep: int) -> Any:
    """
    Truncates target model's KV cache to a specific length of sequence.
    This is necessary when some draft tokens are rejected, so we roll back the cache.
    """
    if past_key_values is None:
        return None
        
    # Hugging Face DynamicCache support
    if hasattr(past_key_values, "crop"):
        past_key_values.crop(num_tokens_to_keep)
        return past_key_values
        
    # Standard tuple/list of tuples format
    # Shape of key/value states: (batch_size, num_heads, seq_len, head_dim)
    new_pkv = []
    for layer in past_key_values:
        k_state, v_state = layer
        k_sliced = k_state[:, :, :num_tokens_to_keep, :]
        v_sliced = v_state[:, :, :num_tokens_to_keep, :]
        new_pkv.append((k_sliced, v_sliced))
        
    return tuple(new_pkv)


def verify_and_accept_greedy(
    draft_tokens: List[int],
    verifier_logits: torch.Tensor
) -> Tuple[List[int], int]:
    """
    Compares proposed draft tokens with verifier logits under greedy decoding.
    Args:
        draft_tokens: list of K proposed token IDs.
        verifier_logits: shape (1, K + 1, vocab_size).
    Returns:
        accepted_tokens: list of token IDs to append to the generation history.
        num_accepted: number of draft tokens accepted (between 0 and K).
    """
    accepted_tokens = []
    num_accepted = 0
    
    # Greedy predictions from verifier
    # verifier_logits shape (1, K + 1, vocab_size)
    verifier_greedy_tokens = torch.argmax(verifier_logits[0], dim=-1).tolist()
    
    for i in range(len(draft_tokens)):
        draft_tok = draft_tokens[i]
        verifier_tok = verifier_greedy_tokens[i]
        
        if draft_tok == verifier_tok:
            accepted_tokens.append(draft_tok)
            num_accepted += 1
        else:
            # Rejection: append the verifier's correction token and stop verification
            accepted_tokens.append(verifier_tok)
            break
    else:
        # If all draft tokens were accepted, we append the next token predicted by the verifier
        accepted_tokens.append(verifier_greedy_tokens[-1])
        
    return accepted_tokens, num_accepted
