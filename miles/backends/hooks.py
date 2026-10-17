"""Small, opt-in backend seams used by shared orchestration.

Native backends retain no-op defaults. Loading a Core hook does not import the
trainer or Open Instruct; optional dependencies are loaded by the operation that
needs them. Keep backend policy and implementations in the backend directory.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from miles.utils.function_registry import load_function


def _configure_rollout(args, user_eval_path: str | None) -> None:
    pass


def _shutdown_rollout(producer) -> None:
    pass


async def _finalize_checkpoint(actor, rollout_id: int) -> None:
    pass


@dataclass(frozen=True)
class BackendHooks:
    # Called after generic path resolution, before checkpoint-eval validation.
    configure_rollout: Callable[[Any, str | None], None] = _configure_rollout
    # Runs synchronously before the executor disposes shared resources.
    shutdown_rollout: Callable[[Any], None] = _shutdown_rollout
    # Awaited after both training and rollout state have been saved.
    finalize_checkpoint: Callable[[Any, int], Awaitable[None]] = _finalize_checkpoint


_DEFAULT_HOOKS = BackendHooks()
_BACKEND_HOOKS = {"olmo_core": "miles.backends.core_utils.integration.get_hooks"}


def get_backend_hooks(backend: str | None) -> BackendHooks:
    path = _BACKEND_HOOKS.get(backend)
    return load_function(path)() if path is not None else _DEFAULT_HOOKS
