from types import SimpleNamespace

import pytest

from miles.utils.arguments import _resolve_rollout_functions, _validate_eval_datasets


def _fully_async_candidate_args(**overrides) -> SimpleNamespace:
    """A namespace shaped like the parsed args `_resolve_rollout_functions` reads."""
    defaults = dict(
        fully_async=False,
        multi_lora=False,
        rollout_function_path=None,
        eval_function_path=None,
        colocate=False,
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
        pause_generation_mode="retract",
        recompute_logprobs_via_prefill=False,
        rollout_all_samples_process_path=None,
        eval_num_gpus=0,
    )
    return SimpleNamespace(**(defaults | overrides))


def test_custom_eval_function_may_own_its_data() -> None:
    args = SimpleNamespace(
        eval_function_path="pkg.CustomEval",
        rollout_function_path="pkg.Rollout",
        eval_datasets=[],
    )

    _validate_eval_datasets(args)


def test_shared_rollout_eval_still_requires_miles_dataset() -> None:
    args = SimpleNamespace(
        eval_function_path="pkg.Rollout",
        rollout_function_path="pkg.Rollout",
        eval_datasets=[],
    )

    with pytest.raises(AssertionError, match="Evaluation datasets must be configured"):
        _validate_eval_datasets(args)


@pytest.mark.parametrize(
    ("mode", "expected_path"),
    [
        ("barrier", "miles.backends.core_utils.rollout.async_rollout.ManagedFullyAsyncRolloutFn"),
        ("refresh", "miles.backends.core_utils.rollout.refreshing_rollout.RefreshingRolloutFn"),
        ("engine_drain", "miles.backends.core_utils.rollout.draining_rollout.DrainingRolloutFn"),
    ],
)
def test_core_publication_mode_selects_matching_rollout_and_eval(mode, expected_path):
    args = _fully_async_candidate_args(
        fully_async=True, train_backend="olmo_core", olmo_core=SimpleNamespace(publication_mode=mode)
    )

    _resolve_rollout_functions(args)

    assert args.rollout_function_path == expected_path
    assert args.eval_function_path == expected_path
    assert not args.eval_uses_snapshots


@pytest.mark.parametrize("mode", ["refresh", "engine_drain"])
def test_core_shared_engine_modes_reject_separate_eval(mode):
    args = _fully_async_candidate_args(
        fully_async=True,
        train_backend="olmo_core",
        olmo_core=SimpleNamespace(publication_mode=mode),
        eval_function_path="custom.eval",
    )

    with pytest.raises(ValueError, match="requires managed shared-engine evaluation"):
        _resolve_rollout_functions(args)
