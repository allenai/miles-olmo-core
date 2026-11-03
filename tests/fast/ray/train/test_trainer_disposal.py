import pytest
import ray
from tests.fast.ray.train.conftest import get_raw_actor_handles
from tests.fast.ray.train.test_group import _make_alive_controller

pytestmark = pytest.mark.asyncio


class TestDispose:
    async def test_dispatches_dispose_to_every_alive_worker(self):
        group = await _make_alive_controller(num_cells=2, actor_count_per_cell=2)

        await group.dispose()

        for cell in group._cells:
            for handle in get_raw_actor_handles(cell):
                calls = ray.get(handle.get_calls.remote())
                assert any(call[0] == "dispose" for call in calls)
