"""Core-only Ulysses boundaries, padding and checkpoint topology contracts."""

from types import SimpleNamespace

import pytest
import torch

from miles.backends.core_utils import context_parallel, contract
from miles.backends.training_utils import parallel
from miles.utils.ft_utils.process_group_utils import GroupInfo

checkpoint = pytest.importorskip("miles.backends.core_utils.checkpoint")


def test_replay_shards_match_core_document_padding_and_tail_counts():
    ring = pytest.importorskip("olmo_core.nn.attention.ring")
    tokens = torch.arange(19).unsqueeze(0)
    batch = {"tokens": tokens, "total_lengths": [7, 12], "doc_lens": torch.tensor([[7, 12]])}
    original = (torch.arange(19).view(1, 19, 1) + torch.arange(2)) % 4
    padded_routes = torch.cat((original, torch.arange(2).expand(1, 13, 2)), dim=1)
    shards = []
    for rank in range(2):
        balancer = ring.UlyssesLoadBalancer(cp_rank=rank, cp_world_size=2)
        model = SimpleNamespace(_cp_load_balancer=balancer)
        result = context_parallel.replay_routes(model, {"router": original}, packed=True)["router"]
        torch.testing.assert_close(result, padded_routes.chunk(2, dim=1)[rank])
        assert bool((result[..., 0] != result[..., 1]).all())
        assert bool(((result >= 0) & (result < 4)).all())
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


def test_cp1_batch_and_scoring_helpers_preserve_pre_forward_validation(monkeypatch):
    monkeypatch.setattr(parallel, "_parallel_state", None)
    monkeypatch.setattr(context_parallel.dist, "get_world_size", lambda: 4)
    assert context_parallel.data_parallel_size() == 4
    assert context_parallel.data_parallel_size(2) == 2
    batch = {"loss_masks": [torch.ones(3)]}
    assert context_parallel.local_masks(batch) is batch["loss_masks"]


def test_cp_pack_builds_host_metadata_without_reading_device_values():
    batch = {"doc_lens": torch.empty((1, 2), dtype=torch.int32, device="meta"), "total_lengths": [7, 12]}
    assert context_parallel.prepare_pack(batch) is batch
    prepared = context_parallel.prepare_pack(batch, 2)
    assert prepared["doc_lens"].device.type == "cpu"
    torch.testing.assert_close(prepared["doc_lens"], torch.tensor([[7, 12]], dtype=torch.int32))
