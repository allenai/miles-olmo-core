"""Core-local per-tensor FP8 export, following the Megatron scaling recipe."""

import torch


def quantize_per_tensor(weight):
    limit = torch.finfo(torch.float8_e4m3fn).max
    # Clamp in FP32 so zero FP16 weights retain a positive scale.
    scale = weight.abs().max().to(torch.float32).clamp(min=1e-12) / limit
    quantized = (weight / scale).clamp(min=-limit, max=limit).to(torch.float8_e4m3fn)
    return quantized, scale.view(1)
