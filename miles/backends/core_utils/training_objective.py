"""Reuse policy-loss scores for Core diagnostics when their distributions match."""

from miles.backends.training_utils.loss import objective


def loss_with_scores(args, batch, count, logits, *, capture, fallback_scores):
    # Core diagnostics currently score the full vocabulary. Support replay scores
    # a restricted distribution; custom/score-centering losses have other APIs.
    # Preserve their existing diagnostic forward until they expose that contract.
    reuse = (
        capture
        and args.loss_type == "policy_loss"
        and batch.get("loss_fn") is None
        and not getattr(args, "use_sampling_support_replay", False)
    )
    scores = [] if reuse else None
    if capture and not reuse:
        batch["training_log_probs"] = fallback_scores(logits, batch)
    result = objective.loss_function(
        args, batch, count, logits, apply_megatron_loss_scaling=False, log_probs_out=scores
    )
    if reuse:
        batch["training_log_probs"] = scores
    return result
