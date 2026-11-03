"""Exercise scoring through the Core actor's shared upstream loss interface."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("olmo_core")

from miles.backends.core_utils import actor
from miles.backends.training_utils import parallel
from miles.utils.ft_utils.process_group_utils import GroupInfo


def test_actor_training_scores_match_direct_response_log_probs(monkeypatch):
    one = GroupInfo(rank=0, size=1, group=None)
    state = parallel.ParallelState(
        intra_dp=one, intra_dp_cp=one, cp=one, tp=one, pp=one, ep=one, etp=one, indep_dp=one
    )
    monkeypatch.setattr(parallel, "_parallel_state", state)
    worker = object.__new__(actor.OLMoCoreTrainRayActor)
    worker.args = SimpleNamespace(
        qkv_format="thd",
        rollout_temperature=1.0,
        true_on_policy_mode=False,
        log_probs_chunk_size=-1,
        allgather_cp=False,
        debug_unified_grad_fused_logprob=False,
    )
    tokens = torch.tensor([0, 1, 2, 3])
    logits = torch.tensor(
        [[[0.1, 0.2, 0.3, 0.4], [0.3, 0.4, 0.2, 0.1], [0.4, 0.2, 0.1, 0.3], [0.2, 0.3, 0.1, 0.4]]], requires_grad=True
    )
    batch = {"unconcat_tokens": [tokens], "total_lengths": [4], "response_lengths": [3], "max_seq_lens": None}
    [scores] = worker._training_log_probs(logits, batch)
    expected = logits.detach()[0, :3].log_softmax(-1).gather(-1, tokens[1:].unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(scores, expected)
    assert not scores.requires_grad
    # Model-training entrypoints must resolve after upstream changed flat modules to packages.
    assert callable(actor.miles_data.get_rollout_data)
    assert callable(actor.miles_loss.compute_advantages_and_returns)
    assert callable(actor.miles_loss.loss_function)
