from __future__ import annotations

from argparse import Namespace

import pytest
from pydantic import ValidationError

from miles.router.config import MilesRouterConfig, compute_miles_router_config


def _make_args(**overrides) -> Namespace:
    defaults = dict(
        miles_router_max_connections=None,
        miles_router_timeout=None,
        rollout_health_check_interval=10.0,
        rollout_health_check_timeout=5.0,
        miles_router_health_check_failure_threshold=3,
        sglang_server_concurrency=64,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


class TestComputeMilesRouterConfig:
    def test_explicit_max_connections_wins(self):
        """--miles-router-max-connections overrides the derived value."""
        config = compute_miles_router_config(
            _make_args(miles_router_max_connections=42), host="10.0.0.1", port=1234, num_engines=4
        )
        assert config.max_connections == 42

    def test_max_connections_derived_from_engine_capacity(self):
        """Without an override, capacity is concurrency * the engines this router fronts,
        which is not the rollout fleet's size when the router fronts the eval fleet."""
        config = compute_miles_router_config(_make_args(), host="10.0.0.1", port=1234, num_engines=4)
        assert config.max_connections == 64 * 4

    def test_remaining_fields_are_copied_from_args(self):
        """Host, port, timeout, and health check settings map one-to-one."""
        config = compute_miles_router_config(
            _make_args(
                miles_router_timeout=30.0,
                rollout_health_check_interval=5.0,
                rollout_health_check_timeout=2.5,
            ),
            num_engines=4,
            host="10.0.0.1",
            port=1234,
        )
        assert config.host == "10.0.0.1"
        assert config.port == 1234
        assert config.timeout == 30.0
        assert config.health_check_interval == 5.0
        assert config.health_check_timeout == 2.5
        assert config.health_check_failure_threshold == 3

    @pytest.mark.parametrize(("num_engines", "capacity"), [(1, 4), (2, 4), (5, 10)])
    def test_health_check_capacity_scales_independently_of_requests(self, num_engines, capacity):
        config = compute_miles_router_config(
            _make_args(miles_router_max_connections=1),
            host="10.0.0.1",
            port=1234,
            num_engines=num_engines,
        )
        assert config.max_connections == 1
        assert config.health_check_max_connections == capacity

    def test_nondefault_health_check_failure_threshold_is_copied(self):
        """A failure threshold other than the default is forwarded, not hardcoded."""
        config = compute_miles_router_config(
            _make_args(miles_router_health_check_failure_threshold=7),
            host="10.0.0.1",
            port=1234,
            num_engines=4,
        )
        assert config.health_check_failure_threshold == 7


_REQUIRED_CONFIG_KWARGS = dict(
    host="127.0.0.1",
    port=1234,
    max_connections=10,
    timeout=5.0,
    health_check_interval=1.0,
    health_check_failure_threshold=3,
)
_COMPLETE_CONFIG_KWARGS = dict(
    **_REQUIRED_CONFIG_KWARGS,
    health_check_timeout=5.0,
    health_check_max_connections=4,
)


class TestMilesRouterConfig:
    def test_complete_kwargs_cover_every_field(self):
        """The per-field omission cases are only meaningful if the full kwargs validate."""
        assert set(_COMPLETE_CONFIG_KWARGS) == set(MilesRouterConfig.model_fields)
        assert MilesRouterConfig(**_COMPLETE_CONFIG_KWARGS).host == "127.0.0.1"

    def test_health_checks_have_defaults_for_existing_callers(self):
        config = MilesRouterConfig(**_REQUIRED_CONFIG_KWARGS)
        assert config.health_check_timeout == 5.0
        assert config.health_check_max_connections == 4

    @pytest.mark.parametrize("missing", sorted(_REQUIRED_CONFIG_KWARGS))
    def test_required_fields_cannot_be_omitted(self, missing: str):
        """Only the health-check timeout and pool capacity have defaults."""
        kwargs = {name: value for name, value in _COMPLETE_CONFIG_KWARGS.items() if name != missing}
        with pytest.raises(ValidationError):
            MilesRouterConfig(**kwargs)
