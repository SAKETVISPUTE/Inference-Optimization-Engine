from typing import List
from src.scheduler.request import Request

class BatchManager:
    def __init__(self, max_batch_size: int = 4):
        self.max_batch_size = max_batch_size
        self.active_requests: List[Request] = []

    def can_accept(self) -> bool:
        return len(self.active_requests) < self.max_batch_size

    def get_free_slots(self) -> int:
        return self.max_batch_size - len(self.active_requests)

    def add_request(self, request: Request):
        if self.can_accept():
            self.active_requests.append(request)
            return True
        return False

    def remove_finished(self) -> List[Request]:
        finished = [req for req in self.active_requests if req.status == "FINISHED"]
        self.active_requests = [req for req in self.active_requests if req.status != "FINISHED"]
        return finished

    def get_active_requests(self) -> List[Request]:
        return self.active_requests
