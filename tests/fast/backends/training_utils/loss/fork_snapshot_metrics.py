"""Adapt only fork diagnostic expectations using archived, validated token scores."""

import torch

ABS_DIFF = "train_rollout_logprob_abs_diff"
TAIL_MAX = f"{ABS_DIFF}_max"
TAIL_FRACTIONS = {f"{ABS_DIFF}_frac_gt_0p5": 0.5, f"{ABS_DIFF}_frac_gt_1p0": 1.0}


def update_current_forward_metrics(args, inputs, saved_outputs):
    """Keep frozen losses/gradients; derive the fork diagnostics from frozen scores.

    Upstream snapshots compare the PPO anchor to rollout scores. The fork reports
    the current forward instead, even when rollout scores are the PPO anchor,
    plus tail statistics (max, fractions above 0.5 and 1.0) of the same masked
    differences. Do not use the current loss implementation to generate these
    expectations.
    """
    if args.loss_type != "policy_loss":
        return
    totals = {ABS_DIFF: [], **{name: [] for name in TAIL_FRACTIONS}, "train_rollout_kl": []}
    maximum = torch.zeros((), dtype=torch.float64)
    for train, rollout, mask in zip(
        saved_outputs["get_log_probs_and_entropy"]["log_probs"],
        inputs["rollout_log_probs"],
        inputs["loss_masks"],
        strict=True,
    ):
        difference = (train.double() - rollout.double()).abs()
        ratio = (train.double() - rollout.double()).clamp(-20, 20)
        values = {
            ABS_DIFF: difference,
            **{name: (difference > edge).double() for name, edge in TAIL_FRACTIONS.items()},
            "train_rollout_kl": (ratio.exp() - 1 - ratio).clamp(-10, 10),
        }
        denominator = 1 if args.calculate_per_token_loss else mask.sum().clamp_min(1)
        for name, value in values.items():
            totals[name].append((value * mask).sum() / denominator)
        maximum = torch.maximum(maximum, (difference * mask).max())
    expected = {name: torch.stack(samples).sum().float() for name, samples in totals.items()}
    expected[TAIL_MAX] = maximum.float()

    saved_outputs["loss_fn"]["metrics"].update(expected)
    dispatcher = saved_outputs["loss_function_dispatcher"]
    keys = list(dispatcher["log_dict_keys"])
    # The leading dispatcher value is the sample/token count, not a metric.
    values = dict(zip(keys, dispatcher["log_dict_values"][1:], strict=True))
    values.update(expected)
    if TAIL_MAX not in keys:
        # The loss reports the tail statistics right after the mean they describe.
        position = keys.index(ABS_DIFF) + 1
        keys[position:position] = [TAIL_MAX, *TAIL_FRACTIONS]
    stored = dispatcher["log_dict_values"]
    dispatcher["log_dict_keys"] = keys
    dispatcher["log_dict_values"] = torch.stack([stored[0], *(values[name].to(stored.dtype) for name in keys)])
