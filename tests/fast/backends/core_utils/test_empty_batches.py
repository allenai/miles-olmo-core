"""Excluded batches preserve optimizer state and distributed participation."""

import contextlib
from datetime import timedelta
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import distributed as dist
from torch import multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

pytest.importorskip("olmo_core")

from miles.backends.core_utils import actor, contract, metrics
from miles.backends.core_utils.publication.state import PolicyClock


def _rollout(mask):
    return {
        "tokens": [torch.tensor([0, 1, 2])],
        "total_lengths": [3],
        "response_lengths": [2],
        "loss_masks": [torch.tensor(mask)],
        "rollout_log_probs": [torch.tensor([-1.0, -1.0])],
        "rewards": [0.0],
        "weight_versions": [[3]],
    }


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        worker = object.__new__(actor.OLMoCoreTrainRayActor)
        worker.args = SimpleNamespace(
            global_batch_size=2, use_rollout_routing_replay=False, save=None,
            olmo_core=SimpleNamespace(max_sequence_length=8, sequence_packing=False, publication_mode="barrier", max_policy_lag=0),
        )
        worker.clock = PolicyClock(completed_steps=3, published_step=3, next_rollout_id=7)
        worker._agree = lambda function: function()
        worker._heartbeat = mock.Mock()
        worker._scoring_pass = mock.Mock(side_effect=AssertionError("Excluded collections must not score"))
        worker._train_steps = mock.Mock(side_effect=AssertionError("Excluded collections must not train"))
        worker.train_module = mock.Mock()
        worker.lr_scheduler = mock.Mock()
        worker.optimizer = torch.optim.AdamW([nn.Parameter(torch.tensor(2.0))], lr=0.1, weight_decay=1.0)
        # Populate optimizer momentum: a zero-gradient step would still move this parameter.
        parameter = worker.optimizer.param_groups[0]["params"][0]
        parameter.grad = torch.tensor(1.0)
        worker.optimizer.step()
        before = parameter.detach().clone()
        momentum = worker.optimizer.state[parameter]["exp_avg"].clone()
        optimizer_step = worker.optimizer.state[parameter]["step"].clone()
        rollout = _rollout([0, 0])
        with mock.patch.object(actor.miles_data, "get_rollout_data", return_value=(rollout, contextlib.nullcontext())), mock.patch.object(metrics.tracking, "log") as log:
            worker.train(7, object())
            assert worker.clock.as_dict() == {"completed_steps": 3, "published_step": 3, "next_rollout_id": 8}
            assert torch.equal(parameter, before)
            assert torch.equal(worker.optimizer.state[parameter]["exp_avg"], momentum)
            assert torch.equal(worker.optimizer.state[parameter]["step"], optimizer_step)
            worker.train_module.zero_grads.assert_not_called()
            worker.lr_scheduler.step.assert_not_called()
            worker._heartbeat.bump.assert_called_once()
            if rank == 0:
                assert log.call_args.kwargs["step_key"] == "train_skips/rollout_id"
                assert log.call_args.args[1]["train_skips/no_active_token_batches"] == 1
            else:
                log.assert_not_called()

        # One rank has no active tokens. Both must take the normal training path.
        worker._scoring_pass = mock.Mock(return_value=SimpleNamespace(standalone=False))
        worker._scoring_check_due = mock.Mock(return_value=False)
        worker._train_steps = mock.Mock()
        worker.ref_module = None
        rollout = _rollout([0, 0] if rank == 0 else [1, 0])
        with mock.patch.object(actor.miles_data, "get_rollout_data", return_value=(rollout, contextlib.nullcontext())), mock.patch.object(metrics.tracking, "log"):
            worker.train(8, object())
        worker._train_steps.assert_called_once()
        [normalization] = worker._train_steps.call_args.args[-1]
        assert normalization.active_tokens == 1
        assert normalization.samples == 2
        assert worker.clock.next_rollout_id == 9

        # Independently check that a masked rank contributes zero, while the
        # surviving response still causes the globally normalized update.
        model = DistributedDataParallel(nn.Linear(1, 1, bias=False))
        with torch.no_grad():
            model.module.weight.fill_(0.5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        prediction = model(torch.tensor([[1.0], [2.0]]))[:, 0]
        losses = (prediction - torch.tensor([1.0, 3.0])).square()
        loss = (losses * rollout["loss_masks"][0]).sum()
        loss = normalization.scale_token_loss(loss)
        loss.backward()
        # One active term divided by the preserved clamped denominator of two.
        torch.testing.assert_close(model.module.weight.grad, torch.tensor([[-0.5]]))
        optimizer.step()
        torch.testing.assert_close(model.module.weight, torch.tensor([[0.55]]))

        # A globally empty step can be identified without weakening the default contract.
        empty = [{"tokens": torch.ones(1, 3), "loss_masks": [torch.zeros(2, dtype=torch.int)]}]
        assert contract.step_normalization(empty, 2, allow_empty=True).active_tokens == 0
        with pytest.raises(ValueError, match="Invalid optimizer batch counts"):
            contract.step_normalization(empty, 2)
        with pytest.raises(ValueError, match="Invalid optimizer batch counts"):
            contract.step_normalization(empty, 4, allow_empty=True)
    finally:
        dist.destroy_process_group()


def test_globally_excluded_and_locally_excluded_ranks(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)


def test_empty_step_inside_active_collection_does_not_update(monkeypatch):
    worker = object.__new__(actor.OLMoCoreTrainRayActor)
    worker.args = SimpleNamespace()
    worker._agree = lambda function: function()
    worker._batch_steps = lambda rollout: [[{"tokens": torch.ones(1, 2)}], [{"tokens": torch.ones(1, 2)}]]
    worker.train_module = mock.Mock()
    worker.lr_scheduler = mock.Mock()
    # Stop at the first active step. An empty step must not access optimizer,
    # expert-packing, schedule, or forward state before this point.
    class ActiveStepReached(Exception):
        pass
    class CoreOptions:
        @property
        def expert_balanced_packing(self):
            raise ActiveStepReached
    worker.args.olmo_core = CoreOptions()
    monkeypatch.setattr(actor.miles_loss, "compute_advantages_and_returns", lambda *args: None)
    monkeypatch.setattr(contract, "validate_training_data", lambda rollout: None)
    monkeypatch.setattr(metrics, "advantage_metrics", lambda rollout: {})
    excluded = contract.StepNormalization(2, 0, 2, 4, 1)
    active = contract.StepNormalization(2, 1, 2, 4, 1)
    worker._batch_steps = lambda rollout: [[{"tokens": torch.ones(1, 2)}]]
    worker._train_steps({}, 0, 2, SimpleNamespace(standalone=True), False, None, None, None, [excluded])
    worker._batch_steps = lambda rollout: [[{"tokens": torch.ones(1, 2)}], [{"tokens": torch.ones(1, 2)}]]
    with pytest.raises(ActiveStepReached):
        worker._train_steps({}, 0, 2, SimpleNamespace(standalone=True), False, None, None, None, [excluded, active])
    worker.train_module.zero_grads.assert_not_called()
    worker.train_module.optim_step.assert_not_called()
    worker.lr_scheduler.step.assert_not_called()
