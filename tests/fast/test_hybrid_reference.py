import torch

from miles_plugins.hybrid.reference import FP32Head


def test_fp32_logits_preserve_stored_weight_dtype_and_gradient():
    linear = torch.nn.Linear(3, 4, bias=False, dtype=torch.bfloat16)
    head = FP32Head(linear)
    inputs = torch.ones(2, 3, dtype=torch.bfloat16, requires_grad=True)
    output = head(inputs)
    assert output.dtype == torch.float32
    assert head.weight is linear.weight
    assert head.weight.dtype == torch.bfloat16
    output.sum().backward()
    assert torch.isfinite(inputs.grad).all()
