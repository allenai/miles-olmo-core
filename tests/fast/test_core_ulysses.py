"""Core-only Ulysses boundaries, padding and checkpoint topology contracts."""

from types import SimpleNamespace

import pytest
import torch

from miles.backends.core_utils import checkpoint, context_parallel, contract
from miles.backends.training_utils import parallel
from miles.utils.ft_utils.process_group_utils import GroupInfo


def test_replay_shards_match_core_document_padding_and_tail_counts():
    ring = pytest.importorskip("olmo_core.nn.attention.ring")
    tokens = torch.arange(19).unsqueeze(0)
    batch = {"tokens": tokens, "total_lengths": [7, 12], "doc_lens": torch.tensor([[7, 12]])}
    original = torch.arange(19 * 2).reshape(1, 19, 2)
    shards = []
    for rank in range(2):
        balancer = ring.UlyssesLoadBalancer(cp_rank=rank, cp_world_size=2)
        model = SimpleNamespace(_cp_load_balancer=balancer)
        result = context_parallel.replay_routes(model, {"router": original}, packed=True)["router"]
        expected = balancer.batch_shard_by_document(
            inputs=[original],
            seq_dims=[1],
            cu_doc_lens=torch.tensor([0, 7, 19]),
            pad_values=[0],
            length_multiple=16,
        )[0][0]
        torch.testing.assert_close(result, expected)
        shards.append(context_parallel.replay_counts(model, batch))
    assert sum(tokens for tokens, _ in shards) == 32
    assert sum(captured for _, captured in shards) == 17


def test_normalization_counts_each_shared_sample_once(monkeypatch):
    monkeypatch.setattr(contract.dist, "all_reduce", lambda value: value.mul_(2))
    monkeypatch.setattr(contract.dist, "get_world_size", lambda: 2)
    batches = [{"tokens": torch.arange(19).unsqueeze(0), "loss_masks": [torch.ones(3), torch.tensor([1, 0, 1])]}]
    result = contract.step_normalization(batches, 2, cp_size=2)
    assert result.samples == 2
    assert result.active_tokens == 5
    assert result.model_tokens == 19
    assert result.scale_token_loss(torch.tensor(5.0)) == 2


def test_checkpoint_rejects_changed_cp_even_with_same_ep_and_world():
    manifest = dict(schema_version=2, world_size=8, expert_parallel_size=8, context_parallel_size=2)
    checkpoint.validate_topology(manifest, 8, 8, 2)
    with pytest.raises(ValueError, match="context_parallel_size"):
        checkpoint.validate_topology(manifest, 8, 8, 1)
    del manifest["context_parallel_size"]
    checkpoint.validate_topology(manifest, 8, 8, 1)
    with pytest.raises(ValueError, match="context_parallel_size"):
        checkpoint.validate_topology(manifest, 8, 8, 2)


def test_cp1_diagnostics_preserve_identity(monkeypatch):
    trivial = GroupInfo(rank=0, size=1, group=None)
    monkeypatch.setattr(
        parallel,
        "_parallel_state",
        parallel.ParallelState(
            intra_dp=trivial,
            intra_dp_cp=trivial,
            cp=trivial,
            tp=trivial,
            pp=trivial,
            ep=trivial,
            etp=trivial,
            indep_dp=trivial,
        ),
    )
    batch = {"tokens": torch.arange(4)}
    assert context_parallel.diagnostic_view(batch) is batch
