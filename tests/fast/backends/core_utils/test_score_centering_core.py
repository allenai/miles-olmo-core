"""Core candidate transport and complete-support score-centering regressions."""

from argparse import Namespace

import pytest
import torch

from miles.backends.core_utils import data, packing
from miles.backends.training_utils.loss.hub.score_centering import ScoreCenteringInputs, score_centering_loss
from miles.utils.score_centering import validate_score_centering_args


@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
def test_complete_support_rounding_does_not_create_a_tail(mode):
    train = torch.tensor([[0.45, 0.55]])
    rollout = torch.tensor([[0.65, 0.35]])
    head = (train * (1 - 2e-6)).log().requires_grad_()
    sampled = head[:, 0]
    inputs = ScoreCenteringInputs(
        train_log_probs=sampled,
        train_head_log_probs=head,
        rollout_log_probs=rollout[:, 0].log(),
        rollout_head_log_probs=rollout.log(),
        head_mask=torch.ones_like(head, dtype=torch.bool),
        advantages=torch.ones(1),
        mode=mode,
        complete_support=True,
    )
    loss, metrics = score_centering_loss(inputs)
    assert torch.equal(metrics.tail_ratio, torch.ones(1))
    # Independent complete-support coefficient: q*f(p/q) - p*f(1).
    p, q = head.detach().exp(), rollout
    ratio = p / q
    weights = torch.ones_like(ratio) if mode == "none" else ratio.clamp(max=2.0)
    if mode == "mis":
        weights = torch.where((ratio >= 0.5) & (ratio <= 5), ratio, 0.0)
    residual = q * weights - p
    expected = -(weights[:, 0] * sampled - (residual * head).sum(-1))
    torch.testing.assert_close(loss, expected)
    actual_grad = torch.autograd.grad(loss.sum(), head, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), head)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def test_candidates_survive_core_sample_and_pack_boundaries():
    rollout = {
        "tokens": [torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5, 6])],
        "total_lengths": [3, 4],
        "response_lengths": [1, 2],
        "loss_masks": [torch.ones(1), torch.ones(2)],
        "rollout_topk_token_ids": [torch.tensor([[2, 4]]), torch.tensor([[5, 6], [6, 7]])],
        "rollout_topk_log_probs": [torch.tensor([[-0.5, -1.0]]), torch.tensor([[-0.4, -1.2], [-0.3, -1.4]])],
    }
    samples = data.sample_batches(rollout, 8)
    packed = packing.combine(samples, [0, 1])
    rebound = packing.bind_training_fields([[packed]], rollout)[0][0]
    for key in ("rollout_topk_token_ids", "rollout_topk_log_probs"):
        for actual, expected in zip(rebound[key], rollout[key], strict=True):
            torch.testing.assert_close(actual, expected)
    assert rebound["response_lengths"] == [1, 2]


def test_score_centering_enables_fp32_rollout_head(monkeypatch):
    monkeypatch.delenv("SGLANG_RETURN_ORIGINAL_LOGPROB", raising=False)
    args = Namespace(
        loss_type="score_centering",
        rollout_top_logprobs_num=20,
        rollout_top_k=-1,
        rollout_temperature=1.0,
        score_centering_tis_clip=2.0,
        score_centering_mis_low=0.5,
        score_centering_mis_high=5.0,
        advantage_estimator="grpo",
        sglang_enable_fp32_lm_head=False,
    )
    validate_score_centering_args(args)
    assert args.sglang_enable_fp32_lm_head is True
