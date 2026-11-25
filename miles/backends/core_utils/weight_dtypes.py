"""Dtype policy for Core weights published to rollout engines."""

import torch

KDA_FP32_WEIGHTS = ("A_log", "dt_bias")


def publication_dtype(name: str, tensor: torch.Tensor) -> torch.dtype:
    # Keep the existing BF16 wire format for weights outside the KDA decay gate.
    return tensor.dtype if name.rsplit(".", 1)[-1] in KDA_FP32_WEIGHTS else torch.bfloat16


def needs_new_bucket(size: int, nbytes: int, dtype: torch.dtype, limit: int) -> bool:
    """SGLang reinterprets a flat byte buffer and requires aligned tensor starts."""
    return size > 0 and (size + nbytes > limit or size % dtype.itemsize != 0)
