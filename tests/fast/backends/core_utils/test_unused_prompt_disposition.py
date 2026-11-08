"""Drop feedback is distinct from consumption, with legacy source compatibility."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from miles.backends.core_utils.rollout import async_rollout


@pytest.mark.parametrize("feedback", [False, True])
@pytest.mark.parametrize("handling", ["drop", "retry"])
def test_unused_handler_preserves_drop_and_retry_contract(monkeypatch, feedback, handling):
    source = SimpleNamespace(acknowledge_groups=Mock(), add_samples=Mock())
    if feedback:
        source.discard_groups = Mock()

    def base_init(self, input):
        self.args, self.data_source = input.args, input.data_source
        self._handle_unused = source.add_samples

    monkeypatch.setattr(async_rollout.FullyAsyncRolloutFn, "__init__", base_init)
    monkeypatch.setattr(async_rollout.async_capacity, "report", lambda *args: {"warnings": []})
    args = SimpleNamespace(async_unused_samples_handler=handling)
    producer = async_rollout.ManagedFullyAsyncRolloutFn(SimpleNamespace(args=args, data_source=source))
    group = [SimpleNamespace(group_index=3)]
    producer._handle_unused(group)
    if handling == "retry":
        source.add_samples.assert_called_once_with(group)
        source.acknowledge_groups.assert_not_called()
        if feedback:
            source.discard_groups.assert_not_called()
    elif feedback:
        source.discard_groups.assert_called_once_with([group])
        source.acknowledge_groups.assert_not_called()
    else:
        source.acknowledge_groups.assert_called_once_with([group])
