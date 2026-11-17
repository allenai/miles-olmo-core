"""Per-tensor serialized FP8 weights shared by trainer exporters."""

import torch


def quantize_per_tensor(weight):
    limit = torch.finfo(torch.float8_e4m3fn).max
    scale = weight.abs().max().to(torch.float32).clamp(min=1e-12) / limit
    quantized = (weight / scale).clamp(min=-limit, max=limit).to(torch.float8_e4m3fn)
    return quantized, scale.view(1)
