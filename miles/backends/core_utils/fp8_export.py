"""Core FP8 export using the same 128x128 CUDA quantizer as Megatron."""

import torch


def quantize_blockwise(weight):
    if weight.ndim != 2 or any(size % 128 for size in weight.shape):
        raise ValueError("Core block FP8 requires expert matrix dimensions divisible by 128")
    if weight.is_cuda:
        # Keep Triton optional for CPU checkpoint conversion and config tests.
        from miles.kernels.quant.fp8_blockwise import fp8_blockwise_cast

        return fp8_blockwise_cast(weight, [128, 128])
    if weight.device.type != "cpu":
        raise ValueError("Core block FP8 export supports CPU or CUDA")
    # CPU reference follows the native kernel: FP32 absmax, epsilon 1e-10,
    # multiplication by reciprocal scale, then E4M3 conversion.
    rows, cols = weight.shape
    blocks = weight.float().reshape(rows // 128, 128, cols // 128, 128)
    scale = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-10) / 448.0
    quantized = (blocks * scale.reciprocal()[:, None, :, None]).clamp(-448.0, 448.0)
    return quantized.reshape(rows, cols).to(torch.float8_e4m3fn), scale
