import torch
import transformers
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
    print("VRAM Available:", torch.cuda.get_device_properties(0).total_memory / (1024**3), "GB")
print("Transformers Version:", transformers.__version__)
