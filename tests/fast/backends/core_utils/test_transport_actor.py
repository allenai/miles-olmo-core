"""Optional Core actor integration; protocol tests do not require OLMo-core."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("olmo_core")

from miles.backends.core_utils import actor


def test_actor_close_delegates_transport_cleanup():
    instance = object.__new__(actor.OLMoCoreTrainRayActor)
    instance.weight_updater = SimpleNamespace(close=Mock())
    instance.close_weight_transport()
    instance.weight_updater.close.assert_called_once_with()
