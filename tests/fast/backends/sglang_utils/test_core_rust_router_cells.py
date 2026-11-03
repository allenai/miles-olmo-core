"""Controller membership is tied to publication, instance identity and health."""

from types import SimpleNamespace
from unittest import mock

import pytest

from miles.backends.sglang_utils import rust_router_membership
from miles.ray.rollout.cell_state import CellAddrInfo, StateDisposed, StatePendingWeights
from miles.ray.rollout.server_cell import ServerCell
from miles.utils.ft_utils.health_checker import ActiveAndEpoch


def make_cell():
    cell = ServerCell.__new__(ServerCell)
    cell.args = SimpleNamespace(
        train_backend="olmo_core",
        use_miles_router=False,
        debug_rollout_only=False,
        rollout_health_check_interval=0,
        rollout_health_check_timeout=1,
        miles_router_health_check_failure_threshold=2,
    )
    cell.meta = SimpleNamespace(update_weights=True, cell_id="test-cell", sglang_api_key=None, worker_type="regular")
    cell.router_api_client = SimpleNamespace(router_url="http://router")
    cell._state = StatePendingWeights(
        addr_info=CellAddrInfo(server_url="http://engine", bootstrap_port=None, gate_url=None)
    )
    cell._rust_membership = mock.Mock(ensure_registered=mock.AsyncMock(), remove=mock.AsyncMock())
    cell._health_checker = mock.Mock()
    cell.health_checker_activeness = lambda: ActiveAndEpoch(active=True, epoch=0)
    return cell


async def test_pending_engine_registers_only_after_publication():
    cell = make_cell()
    try:
        with pytest.raises(RuntimeError, match="confirmed weights"):
            await cell._register_with_router(cell.addr_info)
        await cell.mark_weights_ready()
        assert cell.is_serving
        cell._rust_membership.ensure_registered.assert_awaited_once()
    finally:
        await cell.dispose()


async def test_failed_registration_never_marks_engine_serving():
    cell = make_cell()
    cell._rust_membership.ensure_registered.side_effect = TimeoutError("registration pending")
    try:
        with pytest.raises(TimeoutError):
            await cell.mark_weights_ready()
        assert cell.is_pending_weights
    finally:
        await cell.dispose()


async def test_health_recovery_does_not_readmit_trainable_engine(monkeypatch):
    cell = make_cell()
    await cell.mark_weights_ready()
    health = mock.AsyncMock(side_effect=[False, False, True])
    monkeypatch.setattr(rust_router_membership, "engine_healthy", health)
    try:
        await cell._check_router_membership_health()
        assert cell.is_serving
        await cell._check_router_membership_health()
        assert cell.is_pending_weights
        cell._rust_membership.remove.assert_awaited_once()
        await cell._check_router_membership_health()
        assert cell.is_pending_weights
        assert health.await_count == 2
        assert cell._rust_membership.ensure_registered.await_count == 1
        await cell.mark_weights_ready()
        assert cell.is_serving
        assert cell._rust_membership.ensure_registered.await_count == 2
    finally:
        await cell.dispose()


async def test_failed_quarantine_removal_stays_sticky_even_after_health_recovers(monkeypatch):
    cell = make_cell()
    await cell.mark_weights_ready()
    monkeypatch.setattr(rust_router_membership, "engine_healthy", mock.AsyncMock(return_value=False))
    cell._rust_membership.remove.side_effect = [TimeoutError(), None, None]
    try:
        await cell._check_router_membership_health()
        with pytest.raises(TimeoutError):
            await cell._check_router_membership_health()
        healthy = mock.AsyncMock(return_value=True)
        monkeypatch.setattr(rust_router_membership, "engine_healthy", healthy)
        await cell._check_router_membership_health()
        healthy.assert_not_awaited()
        assert cell.is_pending_weights
    finally:
        await cell.dispose()


async def test_paused_publication_does_not_probe_or_evict(monkeypatch):
    cell = make_cell()
    await cell.mark_weights_ready()
    cell.health_checker_activeness = lambda: ActiveAndEpoch(active=False, epoch=1)
    health = mock.AsyncMock(return_value=False)
    monkeypatch.setattr(rust_router_membership, "engine_healthy", health)
    try:
        for _ in range(5):
            await cell._check_router_membership_health()
        health.assert_not_awaited()
        cell._rust_membership.remove.assert_not_awaited()
        assert cell.is_serving
    finally:
        await cell.dispose()


async def test_disposal_waits_for_pending_membership_and_propagates_failure():
    cell = make_cell()
    cell._rust_membership.remove.side_effect = TimeoutError("removal uncertain")
    with pytest.raises(TimeoutError):
        await cell.dispose()
    assert not isinstance(cell._state, StateDisposed)
    cell._rust_membership.remove.side_effect = None
    await cell.dispose()
    assert isinstance(cell._state, StateDisposed)


@pytest.mark.parametrize("debug_only", [False, True])
async def test_frozen_and_debug_engines_can_recover_without_publication(monkeypatch, debug_only):
    cell = make_cell()
    cell.meta.update_weights = debug_only
    cell.args.debug_rollout_only = debug_only
    monkeypatch.setattr(rust_router_membership, "engine_healthy", mock.AsyncMock(return_value=True))
    try:
        await cell._check_router_membership_health()
        assert cell.is_serving
        cell._rust_membership.ensure_registered.assert_awaited_once()
    finally:
        await cell.dispose()
