"""Excluded responses must not break REINFORCE++ returns or affect active siblings."""

from argparse import Namespace

import pytest
import torch
import torch.distributed as dist
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.training_utils.data.context_parallel import all_gather_with_cp, slice_log_prob_with_cp

from miles.backends.training_utils.loss.hub.advantages import compute_advantages
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state


@pytest.mark.parametrize("gamma", [0.0, 0.9, 1.0])
@pytest.mark.parametrize("kl_coef", [0.0, 0.1])
@pytest.mark.parametrize("excluded_length", [0, 3])
def test_masked_reinforce_returns_preserve_active_responses(gamma, kl_coef, excluded_length):
    group = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(ParallelState(**{key: group for key in ("intra_dp", "intra_dp_cp", "cp", "tp", "pp", "ep", "etp", "indep_dp")}))
    args = Namespace(advantage_estimator="reinforce_plus_plus", gamma=gamma, kl_coef=kl_coef)
    active_kl = torch.tensor([0.2, 0.4, 0.6])
    active_mask = torch.tensor([1.0, 1.0, 0.0])

    def compute(kl, masks, rewards):
        return compute_advantages(
            args=args,
            kl=kl,
            rewards=rewards,
            log_probs=kl,
            loss_masks=masks,
            total_lengths=[len(mask) + 1 for mask in masks],
            response_lengths=[len(mask) for mask in masks],
        )

    _, expected = compute([active_kl.clone()], [active_mask], [1.0])
    advantages, returns = compute(
        [torch.full((excluded_length,), 5.0), active_kl.clone(), torch.ones(2)],
        [torch.zeros(excluded_length), active_mask, torch.zeros(2)],
        [7.0, 1.0, 0.0],
    )
    for values in (advantages, returns):
        torch.testing.assert_close(values[0], torch.zeros(excluded_length))
        torch.testing.assert_close(values[1], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(values[2], torch.zeros(2))


def _cp_worker(rank, world_size, port):
    init_gloo(rank, world_size, port=port)
    try:
        group = GroupInfo(rank=rank, size=world_size, group=dist.group.WORLD)
        trivial = GroupInfo(rank=0, size=1, group=None)
        state = ParallelState(**{key: trivial for key in ("intra_dp", "intra_dp_cp", "cp", "tp", "pp", "ep", "etp", "indep_dp")})
        state.cp = group
        set_parallel_state(state)
        args = Namespace(advantage_estimator="reinforce_plus_plus", gamma=1.0, kl_coef=0.0)
        for total_length, response_length in [(7, 6), (8, 1)]:
            local = slice_log_prob_with_cp(torch.zeros(response_length), total_length, response_length)
            _, returns = compute_advantages(
                args=args,
                kl=[local.clone(), local.clone()],
                rewards=[100.0, 2.0],
                log_probs=[local, local],
                loss_masks=[torch.zeros(response_length), torch.ones(response_length)],
                total_lengths=[total_length] * 2,
                response_lengths=[response_length] * 2,
            )
            excluded = all_gather_with_cp(returns[0], total_length, response_length)
            included = all_gather_with_cp(returns[1], total_length, response_length)
            torch.testing.assert_close(excluded, torch.zeros(response_length))
            torch.testing.assert_close(included, torch.full((response_length,), 2.0))
    finally:
        dist.destroy_process_group()


def test_context_parallel_exclusion_and_empty_local_chunks():
    run_multiprocess(_cp_worker)
