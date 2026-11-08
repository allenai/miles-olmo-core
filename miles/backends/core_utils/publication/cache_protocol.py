"""Fail closed when a policy engine cannot isolate retained prefix state."""

from miles.backends.sglang_utils import sglang_config


def enabled(args):
    if getattr(args, "sglang_enable_policy_versioned_cache", False):
        return True
    if getattr(args, "sglang_config", None) is None:
        return False
    config = sglang_config.resolve_sglang_config(args)
    return any(
        source.update_weights
        and any(
            entry.overrides.get("enable_policy_versioned_cache", False)
            for entry in source.server_groups
            if entry.worker_type != "placeholder"
        )
        for source in config.models
    )


async def read_epoch(engine):
    info = await engine.get_server_info()
    states = info.get("internal_states")
    if not isinstance(states, list) or not states:
        raise RuntimeError("Policy cache requires per-engine internal state readback")
    epochs = []
    for state in states:
        if state.get("disable_radix_cache") is True:
            continue
        cache = state.get("policy_cache")
        if (
            not isinstance(cache, dict)
            or cache.get("protocol") != 1
            or cache.get("updating") is not False
            or type(cache.get("epoch")) is not int
            or cache["epoch"] < 0
        ):
            raise RuntimeError("Inference engine lacks a committed policy-versioned cache")
        epochs.append(cache["epoch"])
    if not epochs:
        return None
    if len(set(epochs)) != 1:
        raise RuntimeError("Policy cache epochs disagree across inference ranks")
    return epochs[0]


async def verify_commit(engine, previous_epoch):
    current = await read_epoch(engine)
    if previous_epoch is None:
        if current is not None:
            raise RuntimeError("Policy cache configuration changed during publication")
    elif current != previous_epoch + 1:
        raise RuntimeError("Weight publication did not advance the policy cache epoch exactly once")
