import torch

from miles.backends.core_utils.weight_dtypes import needs_new_bucket, publication_dtype


def test_kda_publication_preserves_source_dtype():
    fp32 = torch.tensor([1.234567], dtype=torch.float32)
    for name in ("A_log", "dt_bias"):
        key = f"model.layers.0.self_attn.{name}"
        assert publication_dtype(key, fp32) == torch.float32
        assert publication_dtype(key, fp32.bfloat16()) == torch.bfloat16
    assert publication_dtype("model.layers.0.self_attn.q_proj.weight", fp32) == torch.bfloat16


def test_publication_bucket_rolls_over_before_a_misaligned_fp32_weight():
    assert needs_new_bucket(6, 4, torch.float32, 16)
    assert not needs_new_bucket(8, 4, torch.float32, 16)
    assert not needs_new_bucket(6, 4, torch.bfloat16, 16)
    assert needs_new_bucket(8, 12, torch.float32, 16)
    assert not needs_new_bucket(0, 20, torch.float32, 16)
