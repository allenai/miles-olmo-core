import argparse

import pytest
from tests.fast.utils.test_arguments import REQUIRED_ARGS

from miles.utils.arguments import get_miles_extra_args_provider, miles_validate_args


def _validate(*flags, entry="train"):
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)
    args = parser.parse_args(["--train-backend", "olmo_core", "--num-rollout", "1", *REQUIRED_ARGS, *flags])
    args.entry = entry
    miles_validate_args(args)
    return args


@pytest.mark.parametrize(
    "flags",
    [
        ["--rollout-top-k", "32"],
        ["--loss-type", "score_centering", "--rollout-top-k", "32"],
        ["--rollout-top-p", "0.95"],
        ["--rollout-top-p", "0.95", "--rollout-top-k", "32"],
    ],
)
def test_core_policy_rejects_restricted_rollouts(flags):
    with pytest.raises(ValueError, match=r"--rollout-top-p 1 and --rollout-top-k -1") as error:
        _validate(*flags)
    assert "https://github.com/allenai/miles-olmo-core/issues/52" in str(error.value)


@pytest.mark.parametrize("flags", [[], ["--eval-top-p", "0.95", "--eval-top-k", "32"]])
def test_core_policy_accepts_unrestricted_training(flags):
    assert not _validate(*flags).use_sampling_support_replay


def test_rollout_only_can_still_use_restricted_sampling():
    assert _validate(
        "--debug-rollout-only", "--rollout-num-gpus", "1", "--rollout-top-k", "32"
    ).use_sampling_support_replay


def test_serving_can_still_use_restricted_sampling():
    assert _validate("--rollout-top-k", "32", entry="serve").use_sampling_support_replay
