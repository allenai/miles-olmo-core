"""Keep a policy-refresh prompt group when a member fails in transport.

By default (``core.refresh_min_group_size`` unset or equal to
``n_samples_per_prompt``) any recognized serving failure discards the whole
group and requeues its pristine prompts. With a smaller minimum, a member whose
request fails with a transport error or ``refresh_request_timeout`` is replaced
by a fully masked dummy, as long as at least the minimum members succeeded.

The dummy keeps every group, and so every optimizer batch, at its fixed size.
It copies the shortest surviving sibling under the failed member's index, has
a zero loss mask, and is excluded from its group's reward mean and std, so
advantages are computed over the members present. The dynamic filter sees only
the members present. A custom reward post-process (``--custom-reward-post-process-path``)
sees the padded group and must declare ``honors_exclude_from_reward_baseline``.
Group reward models, on-policy distillation and custom train-data conversion
are not supported (``--group-rm``, ``--use-opd`` and
``--custom-convert-samples-to-train-data-path`` are rejected).
"""

import collections
import dataclasses

from miles.backends.core_utils.rollout import rollout_errors
from miles.ray.rollout.train_data_conversion import EXCLUDE_FROM_REWARD_BASELINE, HONORS_EXCLUDE_FROM_REWARD_BASELINE
from miles.utils import logger_utils
from miles.utils.function_registry import load_function
from miles.utils.types import Sample

logger = logger_utils.setup_logger(__name__)

# Failures of one request that say nothing about its prompt. Overload and
# no-worker responses are fleet-wide and keep the whole-group retry path.
DROPPABLE_CATEGORIES = ("transport", "timeout")
DUMMY_KEY = "partial_group_dummy"


def min_group_size(args) -> int | None:
    """Validated minimum successful members per group, or None when partial groups are off."""
    size = getattr(getattr(args, "olmo_core", None), "refresh_min_group_size", None)
    if size is None:
        return None
    samples = args.n_samples_per_prompt
    if type(size) is not int or not 2 <= size <= samples:
        raise ValueError(
            f"core.refresh_min_group_size must be an integer in [2, n_samples_per_prompt={samples}], got {size!r}"
        )
    if size == samples:
        return None
    if getattr(args, "group_rm", False):
        # Group reward models score the whole group before failed members are
        # replaced, so they would see the aborted, incomplete responses.
        raise ValueError("core.refresh_min_group_size below n_samples_per_prompt does not support --group-rm")
    if getattr(args, "use_opd", False):
        # OPD's post-process sets per-row teacher fields, which dummies would lack.
        raise ValueError("core.refresh_min_group_size below n_samples_per_prompt does not support --use-opd")
    if getattr(args, "custom_convert_samples_to_train_data_path", None):
        # A custom converter receives the padded rows and would count the donor copy twice.
        raise ValueError(
            "core.refresh_min_group_size below n_samples_per_prompt does not support "
            "--custom-convert-samples-to-train-data-path"
        )
    post_process = getattr(args, "custom_reward_post_process_path", None)
    if post_process and not getattr(load_function(post_process), HONORS_EXCLUDE_FROM_REWARD_BASELINE, False):
        # The callback sees the padded group; unless it excludes flagged dummies from
        # its statistics, the donor copy would count twice.
        raise ValueError(
            "core.refresh_min_group_size below n_samples_per_prompt requires a custom reward post-process that "
            f"sets {HONORS_EXCLUDE_FROM_REWARD_BASELINE} = True (excludes rows flagged "
            f"{EXCLUDE_FROM_REWARD_BASELINE} from reward statistics and gives them reward 0); got {post_process}"
        )
    return size


def droppable(error: BaseException) -> bool:
    return rollout_errors.recovery_category(error) in DROPPABLE_CATEGORIES


class PartialGroupStats:
    """Cumulative counts, reported with the rollout error metrics."""

    def __init__(self):
        self.accepted_groups = 0
        self.dropped_members = collections.Counter()
        self.requeued_groups = 0

    def metrics(self) -> dict[str, int]:
        return {
            "errors/partial_groups_accepted": self.accepted_groups,
            **{
                f"errors/partial_dropped_{category}_members": self.dropped_members[category]
                for category in DROPPABLE_CATEGORIES
            },
            "errors/partial_requeued_below_min_groups": self.requeued_groups,
        }


def resolve(group: list[Sample], failures: dict[int, BaseException], *, min_size: int, stats) -> list[Sample]:
    """Replace failed members (keyed by ``id(sample)``) by dummies, or raise if too few succeeded."""
    errors = {position: failures.pop(id(sample)) for position, sample in enumerate(group) if id(sample) in failures}
    if not errors:
        return group
    if len(group) - len(errors) < min_size:
        stats.requeued_groups += 1
        raise next(iter(errors.values()))
    survivors = [sample for position, sample in enumerate(group) if position not in errors]
    donor = min(survivors, key=lambda sample: (len(sample.tokens), sample.index))
    categories = {group[position].index: rollout_errors.recovery_category(error) for position, error in errors.items()}
    stats.accepted_groups += 1
    stats.dropped_members.update(categories.values())
    logger.warning(
        "Keeping partial group %s: %d/%d members succeeded, masked dummies for %s",
        group[0].group_index,
        len(survivors),
        len(group),
        categories,
    )
    return [dummy(donor, sample.index) if position in errors else sample for position, sample in enumerate(group)]


def dummy(donor: Sample, index: int) -> Sample:
    return dataclasses.replace(
        donor,
        index=index,
        remove_sample=True,
        loss_mask=[0] * donor.response_length,
        metadata={**donor.metadata, DUMMY_KEY: True, EXCLUDE_FROM_REWARD_BASELINE: True},
    )
