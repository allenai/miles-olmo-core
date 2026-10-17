"""Adapt only fork diagnostic expectations using archived, validated token scores."""

import torch


def update_current_forward_metrics(args, inputs, saved_outputs):
    """Keep frozen losses/gradients; derive the two diagnostics from frozen scores.

    Upstream snapshots compare the PPO anchor to rollout scores. The fork reports
    the current forward instead, even when rollout scores are the PPO anchor.
    Do not use the current loss implementation to generate these expectations.
    """
    if args.loss_type != "policy_loss":
        return
    totals = {"train_rollout_logprob_abs_diff": [], "train_rollout_kl": []}
    for train, rollout, mask in zip(
        saved_outputs["get_log_probs_and_entropy"]["log_probs"],
        inputs["rollout_log_probs"],
        inputs["loss_masks"],
        strict=True,
    ):
        difference = train.double() - rollout.double()
        ratio = difference.clamp(-20, 20)
        values = {
            "train_rollout_logprob_abs_diff": difference.abs(),
            "train_rollout_kl": (ratio.exp() - 1 - ratio).clamp(-10, 10),
        }
        denominator = 1 if args.calculate_per_token_loss else mask.sum().clamp_min(1)
        for name, value in values.items():
            totals[name].append((value * mask).sum() / denominator)
    dispatcher = saved_outputs["loss_function_dispatcher"]
    for name, samples in totals.items():
        expected = torch.stack(samples).sum().float()
        saved_outputs["loss_fn"]["metrics"][name] = expected
        # The leading dispatcher value is the sample/token count, not a metric.
        index = dispatcher["log_dict_keys"].index(name) + 1
        dispatcher["log_dict_values"][index] = expected
