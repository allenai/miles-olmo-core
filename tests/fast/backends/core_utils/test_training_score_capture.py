"""Core score reuse must preserve objective, gradients and diagnostic semantics."""

from unittest.mock import Mock

import pytest
import torch
from tests.fast.backends.training_utils.loss import loss_test_utils as fixtures
from torch import distributed as dist

from miles.backends.core_utils import contract, training_objective
from miles.backends.training_utils import parallel
from miles.backends.training_utils.loss import objective
from miles.backends.training_utils.loss.hub import losses


@pytest.fixture
def runtime(tmp_path):
    previous = parallel._parallel_state
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/group", rank=0, world_size=1)
    fixtures.make_parallel_state()
    try:
        yield
    finally:
        parallel.set_parallel_state(previous)
        dist.destroy_process_group()


def standalone(args, logits, batch):
    with torch.no_grad():
        return objective.get_log_probs_and_entropy(
            logits.detach(),
            args=args,
            unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=batch["total_lengths"],
            response_lengths=batch["response_lengths"],
            max_seq_lens=batch.get("max_seq_lens"),
        )["log_probs"]


@pytest.mark.parametrize("skip_scoring", [False, True])
@pytest.mark.parametrize("token_average", [False, True])
@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("rollout_anchor", [False, True])
@pytest.mark.parametrize("regularization", [False, True])
def test_score_capture_matches_old_two_pass_path(
    runtime, monkeypatch, skip_scoring, token_average, recompute, rollout_anchor, regularization
):
    args = fixtures.make_args(
        true_on_policy_mode=False,
        skip_actor_forward_only=skip_scoring,
        use_rollout_logprobs=rollout_anchor,
        use_tis=not rollout_anchor,
        calculate_per_token_loss=token_average,
        recompute_loss_function=recompute,
        entropy_coef=0.02 if regularization else 0.0,
        observe_training_entropy=False,
        use_kl_loss=regularization,
        kl_loss_coef=0.03 if regularization else 0.0,
        multi_lora=False,
    )
    inputs = fixtures.make_inputs(
        seed=39, batch_size=3, prompt_lens=[2, 3, 4], response_lens=[3, 5, 2], vocab_size=16, args=args
    )
    args.global_batch_size = 3
    inputs["loss_masks"][0][1] = 0
    inputs["loss_masks"][2].zero_()
    baseline_batch = fixtures.make_batch(inputs, "policy_loss")
    candidate_batch = fixtures.make_batch(inputs, "policy_loss")
    baseline_logits = inputs["policy_logits"].clone().requires_grad_()
    candidate_logits = inputs["policy_logits"].clone().requires_grad_()
    expected_scores = standalone(args, baseline_logits, baseline_batch)
    if skip_scoring and not rollout_anchor:
        baseline_batch.pop("log_probs")
        candidate_batch.pop("log_probs")
    baseline, _, baseline_metrics = objective.loss_function(args, baseline_batch, 1, baseline_logits)
    baseline.backward()

    spy = Mock(wraps=losses.get_log_probs_and_entropy)
    monkeypatch.setattr(losses, "get_log_probs_and_entropy", spy)
    fallback = Mock(side_effect=AssertionError("default policy must reuse its loss scores"))
    candidate, _, candidate_metrics = training_objective.loss_with_scores(
        args, candidate_batch, 1, candidate_logits, capture=True, fallback_scores=fallback
    )
    assert spy.call_count == 1
    captured = candidate_batch["training_log_probs"]
    candidate.backward()
    assert spy.call_count == (2 if recompute else 1)
    assert candidate_batch["training_log_probs"] is captured
    assert len(captured) == 3  # recomputation replaces rather than appends
    assert all(not value.requires_grad and value.grad_fn is None for value in captured)
    torch.testing.assert_close(captured, expected_scores, rtol=0, atol=0)
    torch.testing.assert_close(candidate, baseline, rtol=0, atol=0)
    torch.testing.assert_close(candidate_logits.grad, baseline_logits.grad, rtol=0, atol=0)
    assert candidate_metrics["keys"] == baseline_metrics["keys"]
    torch.testing.assert_close(candidate_metrics["values"], baseline_metrics["values"], rtol=0, atol=0)
    assert candidate_logits.grad.count_nonzero() > 0
    sums, maximum = contract.scoring_check(expected_scores, captured, candidate_batch["loss_masks"])
    assert contract.validate_scoring_check(sums, maximum, 1e-3)["mean_abs"] == 0
    drifted = [value + 0.1 for value in captured]
    sums, maximum = contract.scoring_check(expected_scores, drifted, candidate_batch["loss_masks"])
    with pytest.raises(ValueError, match="exceeds tolerance"):
        contract.validate_scoring_check(sums, maximum, 1e-3)


@pytest.mark.parametrize("case", ["standalone", "support_replay", "score_centering", "custom", "tinker"])
def test_other_score_contracts_keep_existing_path(monkeypatch, case):
    args = fixtures.make_args(loss_type="score_centering" if case == "score_centering" else "policy_loss")
    args.use_sampling_support_replay = case == "support_replay"
    if case == "custom":
        args.loss_type = "custom_loss"
    batch = {"loss_fn": "importance_sampling"} if case == "tinker" else {}
    expected = [torch.tensor([0.3])]
    fallback = Mock(return_value=expected)
    dispatch = Mock(return_value=(torch.tensor(0.0), 1, {}))
    monkeypatch.setattr(objective, "loss_function", dispatch)
    training_objective.loss_with_scores(
        args, batch, 1, torch.zeros(1), capture=case != "standalone", fallback_scores=fallback
    )
    assert dispatch.call_args.kwargs["log_probs_out"] is None
    assert fallback.call_count == (0 if case == "standalone" else 1)
    if case != "standalone":
        assert batch["training_log_probs"] is expected


@pytest.mark.parametrize("recompute", [False, True])
def test_diagnostic_scores_survive_in_place_tis_mutation(runtime, monkeypatch, recompute):
    args = fixtures.make_args(
        true_on_policy_mode=False,
        skip_actor_forward_only=True,
        use_tis=True,
        custom_tis_function_path="test.mutating_tis",
        recompute_loss_function=recompute,
        entropy_coef=0.0,
        observe_training_entropy=False,
        use_kl_loss=False,
        multi_lora=False,
    )
    inputs = fixtures.make_inputs(
        seed=39, batch_size=3, prompt_lens=[2, 3, 4], response_lens=[3, 5, 2], vocab_size=16, args=args
    )
    args.global_batch_size = 3
    batch = fixtures.make_batch(inputs, "policy_loss")
    batch.pop("log_probs")
    logits = inputs["policy_logits"].clone().requires_grad_()
    expected = standalone(args, logits, batch)

    def mutating_tis(*, pg_loss, train_log_probs, loss_masks, **kwargs):
        for value in train_log_probs:
            value.add_(1.0)
        return pg_loss, loss_masks, {}

    hook = Mock(side_effect=mutating_tis)
    monkeypatch.setattr(losses, "load_function", lambda path: hook)
    fallback = Mock(side_effect=AssertionError("policy must reuse its loss scores"))
    loss, _, _ = training_objective.loss_with_scores(args, batch, 1, logits, capture=True, fallback_scores=fallback)
    torch.testing.assert_close(batch["training_log_probs"], expected, rtol=0, atol=0)
    loss.backward()
    assert hook.call_count == (2 if recompute else 1)
    torch.testing.assert_close(batch["training_log_probs"], expected, rtol=0, atol=0)
    sums, maximum = contract.scoring_check(expected, batch["training_log_probs"], batch["loss_masks"])
    assert contract.validate_scoring_check(sums, maximum, 1e-3)["mean_abs"] == 0
