import torch
import torch.nn as nn
import torch.nn.functional as F

class INT4Linear(nn.Module):
    """
    Per-channel (per-row) symmetric weight-only INT4 quantized linear layer.
    Packs two 4-bit weights into a single uint8 byte for 8x compression vs FP32.
    Unpacks and dequantizes to floating-point on the fly.
    """
    def __init__(self, in_features, out_features, bias=None, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # We check if in_features is even (required for 4-bit column-wise packing)
        assert in_features % 2 == 0, "in_features must be a multiple of 2 for INT4 packing"
        
        # We store packed weights in uint8. Shape: [out_features, in_features // 2]
        self.register_buffer(
            "weight_packed", 
            torch.zeros((out_features, in_features // 2), dtype=torch.uint8, device=device)
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
    def from_float(cls, float_linear: nn.Linear) -> "INT4Linear":
        """
        Quantizes a floating-point linear layer and packs its weights into INT4 format.
        """
        device = float_linear.weight.device
        in_features = float_linear.in_features
        out_features = float_linear.out_features
        bias = float_linear.bias
        
        # Extract weights in float32 for high-precision scaling calculations
        weight = float_linear.weight.detach().float()
        
        # Per-row symmetric scaling: scale = max(abs(weight)) / 7.0 (signed 4-bit max is 7)
        max_vals = torch.max(torch.abs(weight), dim=1, keepdim=True)[0]
        scale = torch.clamp(max_vals, min=1e-5) / 7.0
        
        # Quantize weight to signed range [-8, 7]
        weight_q = torch.round(weight / scale).clamp(-8, 7).to(torch.int8)
        
        # Shift signed values to unsigned range [0, 15] to prevent sign extension bugs during shifts
        weight_u = (weight_q + 8).to(torch.uint8)
        
        # Extract even and odd columns along the input dimension (columns)
        weight_even = weight_u[:, 0::2]
        weight_odd = weight_u[:, 1::2]
        
        # Pack two 4-bit values into one 8-bit byte:
        # High nibble: even column, Low nibble: odd column
        weight_packed = (weight_even << 4) | weight_odd
        
        # Construct the quantized layer
        instance = cls(in_features, out_features, bias, device=device)
        instance.weight_packed.copy_(weight_packed)
        instance.scale.copy_(scale)
        
        return instance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dynamic unpacking of uint8 packed weights back to signed values in float format
        # 1. Unpack high nibble (even columns) and low nibble (odd columns)
        # Shift right by 4 to get high nibble, and mask with 0x0F (15)
        weight_even = (self.weight_packed >> 4) & 0x0F
        # Mask with 0x0F to get low nibble
        weight_odd = self.weight_packed & 0x0F
        
        # 2. Interleave even and odd back to shape [out_features, in_features]
        out_features, packed_in = self.weight_packed.shape
        in_features = self.in_features
        
        # Create empty tensor on same device and dtype
        weight_u = torch.empty((out_features, in_features), dtype=torch.uint8, device=self.weight_packed.device)
        weight_u[:, 0::2] = weight_even
        weight_u[:, 1::2] = weight_odd
        
        # 3. Convert back to signed range [-8, 7] and cast to input tensor's float dtype
        weight_q = weight_u.to(x.dtype) - 8.0
        
        # 4. Dequantize using the stored scaling factor
        weight_dequant = weight_q * self.scale.to(x.dtype)
        
        # 5. Compute standard linear projection
        return F.linear(x, weight_dequant, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


def quantize_model_to_int4(model: nn.Module) -> nn.Module:
    """
    Recursively replaces all nn.Linear layers in the model with INT4Linear.
    Excludes lm_head to preserve generation accuracy.
    """
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            if name == "lm_head":
                continue
            quantized_layer = INT4Linear.from_float(child)
            setattr(model, name, quantized_layer)
        else:
            quantize_model_to_int4(child)
            
    return model
