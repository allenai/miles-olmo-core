"""Coordinate our cache plugin using the pinned engine's existing API."""

from miles.backends.sglang_utils import sglang_config

BACKEND = "olmo_policy_versioned_v1"


def enabled(args):
    if getattr(args, "sglang_radix_cache_backend", None) == BACKEND:
        return True
    if getattr(args, "sglang_config", None) is None:
        return False
    config = sglang_config.resolve_sglang_config(args)
    return any(
        model.update_weights
        and any(
            group.overrides.get("radix_cache_backend") == BACKEND
            for group in model.server_groups
            if group.worker_type != "placeholder"
        )
        for model in config.models
    )


async def validate_engine(engine, target_version, *, committed):
    info = await engine.get_server_info()
    states = info.get("internal_states")
    if not isinstance(states, list) or not states:
        raise RuntimeError("Policy cache requires per-engine serving state readback")
    for state in states:
        if state.get("disable_radix_cache") is True:
            continue
        if state.get("radix_cache_backend") != BACKEND:
            raise RuntimeError("Retained prefix sharing requires our registered policy-cache backend")
        version = state.get("weight_version")
        if committed:
            if version != str(target_version):
                raise RuntimeError("Weight publication did not commit the expected cache policy version")
        elif version not in (None, "default") and (
            not isinstance(version, str)
            or not version.isascii()
            or not version.isdigit()
            or int(version) > target_version
        ):
            raise RuntimeError("Policy cache requires monotonic nonnegative integer weight versions")
