"""MC score centering: native REINFORCE gradients and the real training dispatcher."""

import numpy as np
import pytest
import torch
from tests.fast.backends.training_utils.test_score_centering_pipeline import single_rank  # noqa: F401
from tests.fast.fixtures.score_centering_fixtures import _args

from miles.backends.training_utils.loss.hub.losses import get_loss_function
from miles.backends.training_utils.loss.hub.score_centering.monte_carlo import monte_carlo_loss


@pytest.mark.parametrize("width", [32, 64, 128])
@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
def test_native_score_gradient_with_repeated_probes(width, mode):
    gen = torch.Generator().manual_seed(17)
    z = torch.randn(4, 11, generator=gen, dtype=torch.float64, requires_grad=True)
    q = torch.randn(4, 11, generator=gen, dtype=torch.float64).log_softmax(-1)
    p = z.log_softmax(-1)
    ids = torch.multinomial(q.exp(), width, replacement=True, generator=gen)
    labels = torch.tensor([0, 1, 2, 3])
    lp, lq = p.gather(1, labels[:, None]).flatten(), q.gather(1, labels[:, None]).flatten()
    probe_p, probe_q = p.gather(1, ids), q.gather(1, ids)
    adv = torch.tensor([1.0, -1.0, 0.4, 0.0], dtype=torch.float64)
    active = torch.tensor([True, True, True, False])
    actual, _ = monte_carlo_loss(
        lp,
        lq,
        adv,
        active,
        probe_logp=probe_p,
        probe_logq=probe_q,
        probe_valid=torch.ones_like(ids, dtype=torch.bool),
        mode=mode,
        mis_low=0.5,
        mis_high=2.0,
    )
    with torch.no_grad():

        def weights(delta):
            if mode == "none":
                return torch.ones_like(delta)
            if mode == "tis":
                return delta.clamp(max=np.log(2.0)).exp()
            return delta.masked_fill((delta < np.log(0.5)) | (delta > np.log(2.0)), -torch.inf).exp()

        sampled_weight = weights(lp - lq)
        probe_weight = weights(probe_p - probe_q)
    correction = (probe_weight * probe_p).mean(-1)
    expected = active * adv * (-sampled_weight * lp + correction)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), z, retain_graph=True)[0], torch.autograd.grad(expected.sum(), z)[0]
    )


def test_masked_rows_can_have_nan_padding_and_bad_active_probes_fail():
    lp = torch.tensor([-0.5, float("nan")], requires_grad=True)
    probes = torch.full((2, 32), -0.7, requires_grad=True)
    q = torch.full_like(probes, -0.8)
    q[1] = float("nan")
    valid = torch.ones_like(probes, dtype=torch.bool)
    valid[1] = False
    loss, _ = monte_carlo_loss(
        lp,
        lp.detach(),
        torch.tensor([1.0, float("nan")]),
        torch.tensor([True, False]),
        probe_logp=probes,
        probe_logq=q,
        probe_valid=valid,
    )
    loss.sum().backward()
    assert torch.isfinite(loss).all() and torch.isfinite(probes.grad).all()
    assert probes.grad[1].count_nonzero() == 0
    valid[0, 0] = False
    with pytest.raises(ValueError, match="Every active"):
        monte_carlo_loss(
            lp,
            lp.detach(),
            torch.ones(2),
            torch.tensor([True, False]),
            probe_logp=probes,
            probe_logq=q,
            probe_valid=valid,
        )


@pytest.mark.usefixtures("single_rank")
def test_mc_dispatcher_scores_real_response_positions():
    args = _args(rollout_score_mode="mc", rollout_top_logprobs_num=32, score_centering_is="mis")
    logits = torch.randn(1, 4, 8, requires_grad=True)
    batch = dict(
        unconcat_tokens=[torch.tensor([0, 1, 2, 3])],
        total_lengths=[4],
        response_lengths=[2],
        loss_masks=[torch.ones(2)],
        rollout_log_probs=[torch.full((2,), -2.0)],
        advantages=[torch.tensor([1.0, -1.0])],
        rollout_topk_token_ids=[np.tile(np.arange(8, dtype=np.int32), (2, 4))],
        rollout_topk_log_probs=[np.full((2, 32), -2.0, dtype=np.float32)],
        rollout_score_mode=["mc"],
    )
    loss, metrics = get_loss_function(args)(args, batch, logits, torch.mean)
    loss.backward()
    assert torch.isfinite(loss) and logits.grad.abs().sum() > 0
    assert "sc_correction" in metrics
    batch["rollout_score_mode"] = ["topk"]
    with pytest.raises(ValueError, match="explicitly marked"):
        get_loss_function(args)(args, batch, logits, torch.mean)
