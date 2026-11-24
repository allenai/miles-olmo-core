"""All data-parallel ranks participate in whitening, including empty shards."""

from argparse import Namespace
from datetime import timedelta

import pytest
import torch
from torch import distributed as dist
from torch import multiprocessing as mp

from miles.backends.training_utils.loss.hub.advantages import normalize_advantages
from miles.backends.training_utils.parallel import ParallelState, set_parallel_state
from miles.utils.ft_utils.process_group_utils import GroupInfo


def _worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=15))
    try:
        local = GroupInfo(rank=0, size=1, group=None)
        dp = GroupInfo(rank=rank, size=2, group=dist.group.WORLD)
        set_parallel_state(
            ParallelState(
                intra_dp=dp, intra_dp_cp=dp, cp=local, tp=local, pp=local, ep=local, etp=local, indep_dp=local
            )
        )

        def whiten(values, mask):
            length = values.numel()
            [result] = normalize_advantages(Namespace(qkv_format="thd"), [values], [mask], [length + 1], [length])
            return result

        values = torch.tensor([1.0, 3.0]) if rank == 0 else torch.empty(0)
        expected = torch.tensor([-1.0, 1.0]) / 2**0.5 if rank == 0 else torch.empty(0)
        torch.testing.assert_close(whiten(values, torch.ones_like(values)), expected)
        dist.barrier()

        # An entirely empty DP group can occur when a context slice owns no response positions.
        torch.testing.assert_close(whiten(torch.empty(0), torch.empty(0)), torch.empty(0))

        # A real response with all positions masked still triggers the existing error.
        with pytest.raises(ValueError, match="global mask sum"):
            whiten(values, torch.zeros_like(values))
    finally:
        dist.destroy_process_group()


def test_advantage_whitening_with_empty_dp_shard(tmp_path):
    mp.spawn(_worker, args=((tmp_path / "whitening-rendezvous").as_uri(),), nprocs=2, join=True)
