import torch
from transformers.cache_utils import DynamicCache

class KVCacheManager:
    """
    Manages Key-Value cache states for autoregressive decoding.
    Wraps Hugging Face DynamicCache for reliability and performance.
    """
    def __init__(self):
        self.cache = DynamicCache()

    def get_cache(self):
        """Returns the underlying cache instance to be passed to the model."""
        return self.cache

    def clear(self):
        """Clears the cached states."""
        self.cache = DynamicCache()

    def get_seq_length(self) -> int:
        """Returns the current sequence length in the cache."""
        if len(self.cache.key_cache) == 0:
            return 0
        return self.cache.get_seq_length()

    def get_memory_footprint_bytes(self) -> int:
        """
        Calculates the memory footprint of the cached Key and Value tensors in bytes.
        """
        total_bytes = 0
        for layer_k in self.cache.key_cache:
            total_bytes += layer_k.numel() * layer_k.element_size()
        for layer_v in self.cache.value_cache:
            total_bytes += layer_v.numel() * layer_v.element_size()
        return total_bytes
