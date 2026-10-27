from types import SimpleNamespace

import torch

from miles_plugins.hybrid.reference import FP32Head, cached_logprobs


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


def test_cached_reference_aligns_prompt_suffix_and_continuation():
    class TokenOracle:
        device = "cpu"

        def __init__(self):
            self.calls = []

        def __call__(self, inputs, past_key_values=None, use_cache=False, logits_to_keep=1):
            self.calls.append((inputs.tolist(), past_key_values, use_cache))
            logits = torch.nn.functional.one_hot(inputs, num_classes=8).float()
            return SimpleNamespace(logits=logits[:, -logits_to_keep:], past_key_values="cached")

    model = TokenOracle()
    scores = cached_logprobs(model, [1, 2, 3, 4], [5, 6, 7], first_position=2)
    # Predict target positions 2..6: prompt predecessors 2,3,4 then generated 5,6.
    assert scores.argmax(-1).tolist() == [2, 3, 4, 5, 6]
    assert model.calls == [([[1, 2, 3, 4]], None, True), ([[5]], "cached", True), ([[6]], "cached", True)]
