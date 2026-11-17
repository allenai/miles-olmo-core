"""Own publication buffers without losing serialized quantization metadata."""

import torch


def nbytes(tensor, *, quantized):
    return tensor.nbytes if quantized else tensor.numel() * 2


def requires_flush(size, tensor, *, quantized, limit):
    itemsize = tensor.element_size() if quantized else 2
    return bool(size and (size + nbytes(tensor, quantized=quantized) > limit or size % itemsize))


def copy_tensor(tensor, *, quantized, device=None):
    return (
        tensor.detach()
        .to(
            device=device or tensor.device,
            dtype=tensor.dtype if quantized else torch.bfloat16,
            copy=True,
        )
        .contiguous()
    )
