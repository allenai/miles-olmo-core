"""Token-loss denominators at the Core boundary: the step's active tokens, or samples x a fixed length."""

import pytest
import torch

from miles.backends.core_utils import contract


def normalization(constant=None):
    return contract.StepNormalization(
        samples=4,
        active_tokens=30,
        token_denominator=30,
        model_tokens=60,
        world_size=2,
        constant_token_denominator=constant,
    )


def test_default_divides_by_active_tokens():
    loss = torch.tensor(12.0)
    assert normalization().token_loss_denominator == 30
    torch.testing.assert_close(normalization().scale_token_loss(loss), loss * 2 / 30)


def test_constant_divides_by_samples_times_length():
    loss = torch.tensor(12.0)
    norm = normalization(constant=16)
    assert norm.token_loss_denominator == 4 * 16
    torch.testing.assert_close(norm.scale_token_loss(loss), loss * 2 / 64)


@pytest.mark.parametrize("active", [10, 30, 50])
def test_constant_ignores_step_length(active):
    norm = contract.StepNormalization(4, active, active, 2 * active, 2, constant_token_denominator=16)
    assert norm.token_loss_denominator == 64
