import torch
import torch.nn as nn
import torch.nn.functional as F

class INT8Linear(nn.Module):
    """
    Per-channel (per-row) symmetric weight-only INT8 quantized linear layer.
    Stores weights in int8, scales in float32, and dequantizes on the fly.
    """
    def __init__(self, in_features, out_features, bias=None, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Buffers are non-parameter tensors that are part of the module state
        self.register_buffer(
            "weight_q", 
            torch.zeros((out_features, in_features), dtype=torch.int8, device=device)
        )
        self.register_buffer(
            "scale", 
            torch.zeros((out_features, 1), dtype=torch.float32, device=device)
        )
        
        if bias is not None:
            self.bias = nn.Parameter(bias.clone().to(device))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_float(cls, float_linear: nn.Linear) -> "INT8Linear":
        """
        Quantizes a floating-point linear layer into INT8 Linear.
        """
        device = float_linear.weight.device
        in_features = float_linear.in_features
        out_features = float_linear.out_features
        bias = float_linear.bias
        
        # Extract weights and move to float32 for scaling calculations
        weight = float_linear.weight.detach().float()
        
        # Per-row symmetric scaling: scale = max(abs(weight)) / 127
        max_vals = torch.max(torch.abs(weight), dim=1, keepdim=True)[0]
        # Avoid division by zero
        scale = torch.clamp(max_vals, min=1e-5) / 127.0
        
        # Round and quantize weights
        weight_q = torch.round(weight / scale).to(torch.int8)
        
        # Construct the quantized layer
        instance = cls(in_features, out_features, bias, device=device)
        instance.weight_q.copy_(weight_q)
        instance.scale.copy_(scale)
        
        return instance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dynamic dequantization of weight to match input tensor dtype (float32/float16/bfloat16)
        weight_dequant = self.weight_q.to(x.dtype) * self.scale.to(x.dtype)
        return F.linear(x, weight_dequant, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


def quantize_model_to_int8(model: nn.Module) -> nn.Module:
    """
    Recursively replaces all nn.Linear layers in the model with INT8Linear.
    """
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            # Check if this is a standard linear layer (e.g. QKV projection, MLP projections)
            # Skip lm_head to avoid severe quality loss (standard practice in LLM quantization)
            if name == "lm_head":
                continue
            
            # Replace child linear layer
            quantized_layer = INT8Linear.from_float(child)
            setattr(model, name, quantized_layer)
        else:
            # Recurse
            quantize_model_to_int8(child)
            
    return model
