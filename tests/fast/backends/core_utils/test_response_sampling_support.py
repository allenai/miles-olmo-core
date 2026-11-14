"""Core score anchors must use the same sampler support as the policy loss."""

from types import SimpleNamespace

import pytest
import torch

from miles.backends.core_utils import data, packing, response_scoring
from miles.backends.training_utils.loss.hub import logit_processors


@pytest.mark.parametrize("support_replay", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_core_scores_match_independent_categorical_reference(monkeypatch, support_replay, packed):
    state = SimpleNamespace(tp=SimpleNamespace(rank=0, group=None), cp=SimpleNamespace(rank=0, size=1))
    monkeypatch.setattr(logit_processors, "get_parallel_state", lambda: state)
    args = SimpleNamespace(
        use_sampling_support_replay=support_replay,
        qkv_format="thd",
        rollout_temperature=1.0,
        true_on_policy_mode=True,
        bf16=False,
        fp16=False,
        log_probs_chunk_size=-1,
        vocab_size=4,
        allgather_cp=False,
        debug_unified_grad_fused_logprob=False,
    )
    rollout = {
        "tokens": [torch.tensor([2, 0, 3]), torch.tensor([1, 2])],
        "total_lengths": [3, 2],
        "response_lengths": [2, 1],
        "loss_masks": [torch.ones(2), torch.ones(1)],
        "rollout_sampling_mask_ids": [
            torch.tensor([0, 2, 1, 3], dtype=torch.int32),
            torch.tensor([0, 2], dtype=torch.int32),
        ],
        "rollout_sampling_mask_offsets": [torch.tensor([0, 2, 4]), torch.tensor([0, 2])],
    }
    logits = [
        torch.tensor([[[2.0, 1.0, 0.0, -1.0], [-1.0, 0.0, 1.0, 2.0], [0.0, 0.0, 0.0, 0.0]]]),
        torch.tensor([[[1.0, 0.0, -2.0, -1.0], [0.0, 0.0, 0.0, 0.0]]]),
    ]
    batches = data.sample_batches(rollout, 8)
    if packed:
        batches = [packing.combine(batches, [0, 1])]
        logits = [torch.cat(logits, dim=1)]
    scores = [
        score
        for batch, values in zip(batches, logits, strict=True)
        for score in response_scoring.log_probs(args, values, batch)
    ]
    selected = [
        (torch.tensor([2.0, 1.0, 0.0, -1.0]), 0, [0, 2]),
        (torch.tensor([-1.0, 0.0, 1.0, 2.0]), 3, [1, 3]),
        (torch.tensor([1.0, 0.0, -2.0, -1.0]), 2, [0, 2]),
    ]
    expected = []
    for row, token, support in selected:
        reference = row[support] if support_replay else row
        target = support.index(token) if support_replay else token
        expected.append(torch.log_softmax(reference, dim=-1)[target])
    torch.testing.assert_close(torch.cat(scores), torch.stack(expected))


def test_support_replay_refuses_missing_masks_before_scoring():
    args = SimpleNamespace(use_sampling_support_replay=True)
    with pytest.raises(ValueError):
        response_scoring.log_probs(args, torch.zeros(1, 2, 4), {})
