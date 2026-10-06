import pytest


def test_bridge_plugins_retry_after_megatron_initialization() -> None:
    pytest.importorskip("megatron.bridge", reason="Megatron Bridge is not installed in CPU CI")

    from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping

    from miles.backends.megatron_utils import model_provider  # noqa: F401
    from miles_plugins import megatron_bridge as megatron_bridge_plugin
    from miles_plugins.megatron_bridge import nemotron_h

    assert getattr(MegatronParamMapping, "_miles_pp_group_unwrap_installed", False)
    assert nemotron_h._MILES_NEMOTRON_H_BRIDGE_INSTALLED

    megatron_bridge_plugin.install()

    assert MegatronParamMapping._miles_pp_group_unwrap_installed
    assert nemotron_h._MILES_NEMOTRON_H_BRIDGE_INSTALLED
