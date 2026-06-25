import time
from typing import List, Optional

class Request:
    def __init__(
        self,
        request_id: str,
        prompt: str,
        prompt_ids: List[int],
        max_new_tokens: int = 128
    ):
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_ids = prompt_ids
        self.max_new_tokens = max_new_tokens
        
        # Generation state
        self.generated_ids: List[int] = []
        self.status = "PENDING"  # PENDING, RUNNING, FINISHED
        
        # Timing metrics
        self.arrival_time: float = time.time()
        self.start_time: Optional[float] = None
        self.finish_time: Optional[float] = None
        
        # Iteration tracking
        self.num_steps = 0

    @property
    def total_tokens(self) -> int:
        return len(self.prompt_ids) + len(self.generated_ids)

    def start(self):
        self.status = "RUNNING"
        self.start_time = time.time()

    def add_token(self, token_id: int):
        self.generated_ids.append(token_id)
        self.num_steps += 1

    def finish(self):
        self.status = "FINISHED"
        self.finish_time = time.time()
