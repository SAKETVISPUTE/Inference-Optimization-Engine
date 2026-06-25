import os
import torch
import gc

class GPUProfiler:
    """
    Context manager that wraps torch.profiler to capture GPU and CPU traces.
    Saves results as Chrome Tracing JSON format (compatible with Perfetto/chrome://tracing).
    """
    def __init__(self, name: str, output_dir: str = "reports/traces"):
        self.name = name
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.prof = None

    def __enter__(self):
        # Force garbage collection and clear CUDA cache to ensure clean profiling
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
        # Initialize PyTorch Profiler
        self.prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ] if torch.cuda.is_available() else [torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        self.prof.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.prof:
            self.prof.stop()
            trace_path = os.path.join(self.output_dir, f"{self.name}_trace.json")
            self.prof.export_chrome_trace(trace_path)
            print(f"Profile saved: Chrome trace exported to {trace_path}")


def get_cuda_memory_allocated_mb() -> float:
    """Returns currently allocated CUDA memory in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0


def get_cuda_max_memory_allocated_mb() -> float:
    """Returns peak allocated CUDA memory in MB since start or last reset."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def reset_cuda_memory_stats():
    """Resets peak CUDA memory tracking stats."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
