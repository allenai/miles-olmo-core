"""Score Core responses under the distribution used by the rollout sampler."""

from miles.backends.training_utils.data import sampling_mask
from miles.backends.training_utils.loss import objective


def log_probs(args, logits, batch):
    supports = sampling_mask.get_rollout_sampling_masks(batch) if getattr(args, "use_sampling_support_replay", False) else None
    return objective.get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch["max_seq_lens"],
        rollout_sampling_mask=supports,
    )["log_probs"]
