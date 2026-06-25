from typing import List
from src.scheduler.request import Request

class RequestQueue:
    def __init__(self):
        self.queue: List[Request] = []

    def add_request(self, request: Request):
        self.queue.append(request)

    def has_pending(self) -> bool:
        return any(req.status == "PENDING" for req in self.queue)

    def get_pending_requests(self) -> List[Request]:
        return [req for req in self.queue if req.status == "PENDING"]

    def pop_pending(self, limit: int = 1) -> List[Request]:
        pending = self.get_pending_requests()
        selected = pending[:limit]
        for req in selected:
            req.start()
        return selected

    def all_finished(self) -> bool:
        return all(req.status == "FINISHED" for req in self.queue)

    def get_all_requests(self) -> List[Request]:
        return self.queue
