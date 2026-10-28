"""An uncertain asynchronous registration must remain owned across disposal retries."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from tests.fast.ray.rollout.test_server_cell_dispose import _make_cell, _make_router_api_client

from miles.backends.sglang_utils.router_registration import RegistrationCleanupError
from miles.ray.rollout.cell_state import CellAddrInfo, StateDisposed, StateInitializing, StatePendingWeights
from miles.ray.rollout.weight_admission import defer_weight_admission


@pytest.mark.asyncio
@pytest.mark.parametrize("initializing", [False, True])
async def test_uncertain_cleanup_retains_cell_until_a_later_dispose_confirms_removal(initializing):
    api = _make_router_api_client()
    api.add_worker = AsyncMock(side_effect=asyncio.CancelledError())
    api.remove_worker = AsyncMock(side_effect=RegistrationCleanupError("job still pending"))
    cell = _make_cell(router_api_client=api)
    addr = CellAddrInfo(server_url="http://engine:30000", bootstrap_port=None, gate_url=None)
    cell._state = (
        StateInitializing(addr_info=addr, start_time=0) if initializing else StatePendingWeights(addr_info=addr)
    )
    with pytest.raises(asyncio.CancelledError):
        await cell._register_with_router(addr)
    assert cell._registration_attempted
    with pytest.raises(RegistrationCleanupError):
        await cell.dispose()
    assert not isinstance(cell._state, StateDisposed)
    assert cell._registration_attempted
    api.remove_worker = AsyncMock()
    await cell.dispose()
    assert isinstance(cell._state, StateDisposed)
    assert not cell._registration_attempted


@pytest.mark.asyncio
async def test_abort_failure_does_not_replace_the_publication_error():
    controller = SimpleNamespace(
        begin_weight_admission=AsyncMock(),
        finish_weight_admission=AsyncMock(side_effect=RuntimeError("abort RPC failed")),
    )
    with pytest.raises(ValueError, match="diagnostics failed"):
        async with defer_weight_admission(controller):
            raise ValueError("diagnostics failed")
    controller.finish_weight_admission.assert_awaited_once_with(admit=False)
