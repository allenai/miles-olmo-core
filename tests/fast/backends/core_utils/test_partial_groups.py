"""Partial policy-refresh groups: a transport-failed member becomes a fully masked dummy."""

import asyncio
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import torch
from tests.fast.ray.rollout.conftest import make_args

from miles.backends.core_utils import data as core_data
from miles.backends.core_utils import observability, packing
from miles.backends.core_utils.publication import policy_refresh
from miles.backends.core_utils.rollout import partial_groups, rollout_errors
from miles.backends.core_utils.rollout.async_buffer import RefreshPolicyDataBuffer
from miles.backends.core_utils.rollout.errors import GenerationRequestTimeout
from miles.backends.core_utils.rollout.refreshing_rollout import RefreshingRolloutFn
from miles.ray.rollout import metrics as rollout_metrics
from miles.ray.rollout import train_data_conversion
from miles.rollout.base_types import GenerateFnOutput
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput
from miles.utils.inference_records import response_record
from miles.utils.types import PARTIAL_GROUP_PADDING, Sample


def router_failure(code="call_upstream_request_error"):
    request = httpx.Request("POST", "http://router/generate")
    response = httpx.Response(500, request=request, json={"error": {"code": code}})
    return httpx.HTTPStatusError("router failure", request=request, response=response)


def complete(sample, *, reward=1.0, response_length=1):
    sample.response = "answer"
    sample.tokens = [9] + [10] * response_length
    sample.response_length = response_length
    sample.rollout_log_probs = [-0.8] * response_length
    sample.reward = reward
    sample.status = Sample.Status.COMPLETED
    policy_refresh.record_response(
        sample,
        dict(
            weight_version="0",
            weight_versions=[dict(version="0", start=0, end=response_length)],
            output_token_logprobs=[[-0.8, 10]] * response_length,
        ),
    )
    return sample


def make_sample(group_index, index, **kwargs):
    return complete(Sample(index=index, group_index=group_index, prompt="question"), **kwargs)


def test_min_group_size_defaults_to_full_group_and_validates():
    def args(size):
        return SimpleNamespace(n_samples_per_prompt=8, olmo_core=SimpleNamespace(refresh_min_group_size=size))

    assert partial_groups.min_group_size(SimpleNamespace(olmo_core=SimpleNamespace())) is None
    assert partial_groups.min_group_size(args(None)) is None
    assert partial_groups.min_group_size(args(8)) is None
    assert partial_groups.min_group_size(args(7)) == 7
    for invalid in (1, 9, 7.0, True):
        with pytest.raises(ValueError, match="refresh_min_group_size"):
            partial_groups.min_group_size(args(invalid))
    with pytest.raises(ValueError, match="group-rm"):
        partial_groups.min_group_size(SimpleNamespace(**vars(args(7)), group_rm=True))
    assert partial_groups.min_group_size(SimpleNamespace(**vars(args(8)), group_rm=True)) is None
    # OPD post-processing and custom converters would see dummy rows; refuse them at startup.
    with pytest.raises(ValueError, match="use-opd"):
        partial_groups.min_group_size(SimpleNamespace(**vars(args(7)), use_opd=True))
    with pytest.raises(ValueError, match="custom-convert"):
        partial_groups.min_group_size(
            SimpleNamespace(**vars(args(7)), custom_convert_samples_to_train_data_path="custom.convert")
        )
    assert partial_groups.min_group_size(SimpleNamespace(**vars(args(7)), use_opd=False)) == 7
    assert (
        partial_groups.min_group_size(
            SimpleNamespace(**vars(args(8)), use_opd=True, custom_convert_samples_to_train_data_path="x")
        )
        is None
    )


def failed_group(failed, lengths=None):
    group = [make_sample(3, i, response_length=(lengths or {}).get(i, 3)) for i in range(8)]
    failures = {}
    for i in failed:
        group[i].status = Sample.Status.ABORTED
        failures[id(group[i])] = router_failure() if i % 2 else GenerationRequestTimeout("timeout")
    return group, failures


def test_seven_of_eight_keeps_group_with_masked_dummy():
    group, failures = failed_group([2], lengths={5: 1})
    stats = partial_groups.PartialGroupStats()
    resolved = partial_groups.resolve(group, failures, min_size=7, stats=stats)
    dummy = resolved[2]
    assert failures == {}
    assert [s.index for s in resolved] == list(range(8))
    assert resolved[:2] + resolved[3:] == group[:2] + group[3:]
    assert (dummy.group_index, dummy.response_length, dummy.tokens) == (3, 0, [0])
    assert dummy.remove_sample and dummy.loss_mask == []
    assert dummy.response == "" and dummy.rollout_log_probs == []
    assert dummy.weight_versions == [] and dummy.train_metadata == {PARTIAL_GROUP_PADDING: True}
    assert dummy.status == Sample.Status.FAILED
    record = response_record(dummy, text=True, fresh_run=True)
    assert record["reward"] is None and record["response_tokens"] == 0 and "response" not in record
    assert record[PARTIAL_GROUP_PADDING]
    assert dummy.metadata[PARTIAL_GROUP_PADDING]
    assert not group[5].metadata.get(PARTIAL_GROUP_PADDING)
    assert (stats.accepted_groups, stats.dropped_members["timeout"], stats.requeued_groups) == (1, 1, 0)


def test_six_of_eight_requeues_the_group_when_min_is_seven():
    group, failures = failed_group([1, 2])
    stats = partial_groups.PartialGroupStats()
    with pytest.raises(httpx.HTTPStatusError):
        partial_groups.resolve(group, failures, min_size=7, stats=stats)
    assert failures == {}
    assert (stats.accepted_groups, sum(stats.dropped_members.values()), stats.requeued_groups) == (0, 0, 1)
    assert len(partial_groups.resolve(*failed_group([1, 2]), min_size=6, stats=stats)) == 8


def buffer_args(min_size):
    return SimpleNamespace(
        rollout_global_dataset=True,
        rollout_batch_size=1,
        n_samples_per_prompt=8,
        global_batch_size=8,
        max_weight_staleness=1,
        async_data_buffer_capacity_factor=4,
        dynamic_sampling_filter_path=None,
        reward_key=None,
        olmo_core=SimpleNamespace(refresh_min_group_size=min_size),
    )


def test_refresh_buffer_accepts_the_seven_member_group():
    async def scenario():
        buffer = RefreshPolicyDataBuffer(DataBufferConstructorInput(buffer_args(7), lambda group: None))
        group, failures = failed_group([4])
        resolved = partial_groups.resolve(group, failures, min_size=7, stats=partial_groups.PartialGroupStats())
        assert await buffer.put(DataBufferInput(prompt_group=group, group=resolved)) is True
        assert len(buffer._delegate._buffer) == 1

    asyncio.run(scenario())


def reward_args():
    return SimpleNamespace(
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=True,
        n_samples_per_prompt=8,
        rollout_batch_size=2,
        reward_key=None,
        rollout_top_logprobs_num=0,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        use_dynamic_global_batch_size=False,
    )


def reference(rewards, std_normalization=True):
    values = torch.tensor(rewards, dtype=torch.float)
    centered = values - values.mean()
    if std_normalization and values.std() > 0:
        centered = centered / (values.std() + 1e-6)
    return centered.tolist()


@pytest.mark.parametrize("std_normalization", [True, False])
def test_variable_group_size_advantages_match_reference(std_normalization):
    full_rewards = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    partial_rewards = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    full = [make_sample(0, i, reward=r) for i, r in enumerate(full_rewards)]
    partial = [make_sample(1, 8 + i, reward=r) for i, r in enumerate(partial_rewards)]
    failures = {id(partial[2]): router_failure()}
    partial = partial_groups.resolve(partial, failures, min_size=7, stats=partial_groups.PartialGroupStats())
    args = reward_args()
    args.grpo_std_normalization = std_normalization

    _, normalized = train_data_conversion._post_process_rewards(args, full + partial, None)

    present = partial_rewards[:2] + partial_rewards[3:]
    assert normalized[:8] == pytest.approx(reference(full_rewards, std_normalization))
    expected = reference(present, std_normalization)
    assert normalized[8:10] + normalized[11:] == pytest.approx(expected)
    # The dummy neither moves its group's baseline nor gets an advantage.
    assert normalized[10] == 0.0


def test_unflagged_batches_normalize_exactly_as_before():
    rewards = [1.0, 0.0, 0.5, 1.0, 0.0, 0.0, 1.0, 0.25] * 2
    samples = [make_sample(i // 8, i, reward=r) for i, r in enumerate(rewards)]
    _, normalized = train_data_conversion._post_process_rewards(reward_args(), samples, None)
    assert normalized == reference(rewards[:8]) + reference(rewards[8:])


def mean_centering(args, samples):
    """A third-party post-process that centers rewards per group and knows nothing of dummies."""
    raw = [sample.get_reward_value(args) for sample in samples]
    return raw, [reward - sum(raw) / len(raw) for reward in raw]


def baseline_aware(args, samples):
    """A post-process that declares it excludes flagged rows from its statistics."""
    raw = [sample.get_reward_value(args) for sample in samples]
    present = [
        r for r, s in zip(raw, samples, strict=True) if not train_data_conversion.excluded_from_reward_baseline(s)
    ]
    mean = sum(present) / len(present)
    return raw, [
        0.0 if train_data_conversion.excluded_from_reward_baseline(s) else r - mean
        for r, s in zip(raw, samples, strict=True)
    ]


baseline_aware.honors_exclude_from_reward_baseline = True


def test_partial_groups_require_a_baseline_aware_custom_post_process():
    # A callback that does not exclude padding would count its placeholder reward, and one
    # that reshapes by n_samples_per_prompt cannot see prompt boundaries otherwise.
    args = SimpleNamespace(n_samples_per_prompt=8, olmo_core=SimpleNamespace(refresh_min_group_size=7))
    path = f"{__name__}.mean_centering"
    with pytest.raises(ValueError, match="honors_exclude_from_reward_baseline"):
        partial_groups.min_group_size(SimpleNamespace(**vars(args), custom_reward_post_process_path=path))
    aware = f"{__name__}.baseline_aware"
    assert partial_groups.min_group_size(SimpleNamespace(**vars(args), custom_reward_post_process_path=aware)) == 7
    # Full groups keep every callback.
    args.olmo_core.refresh_min_group_size = 8
    assert partial_groups.min_group_size(SimpleNamespace(**vars(args), custom_reward_post_process_path=path)) is None


def test_baseline_aware_post_process_sees_the_padded_group():
    # Survivors 0 and 1 center to -0.5 and 0.5; counting the zero placeholder would give -1/3 and 2/3.
    group = [make_sample(0, 0, reward=0.0, response_length=1), make_sample(0, 1, reward=1.0, response_length=2)]
    group.append(make_sample(0, 2, reward=1.0))
    resolved = partial_groups.resolve(
        group, {id(group[2]): router_failure()}, min_size=2, stats=partial_groups.PartialGroupStats()
    )
    data = train_data_conversion.convert_samples_to_train_data(reward_args(), resolved, {}, None, baseline_aware)
    assert data["rewards"] == pytest.approx([-0.5, 0.5, 0.0])
    assert data["loss_masks"][2] == []


def test_filters_and_reward_metrics_see_only_the_members_present(monkeypatch):
    seen = []

    def dynamic_filter(args, group, **kwargs):
        seen.append([sample.index for sample in group])
        return SimpleNamespace(keep=True, reason=None)

    async def scenario():
        buffer = RefreshPolicyDataBuffer(DataBufferConstructorInput(buffer_args(7), lambda group: None))
        measured = buffer._delegate
        monkeypatch.setattr(measured, "_group_filter", dynamic_filter)
        group = [make_sample(0, i, reward=float(i == 0), response_length=1 + i) for i in range(8)]
        resolved = partial_groups.resolve(
            group, {id(group[4]): router_failure()}, min_size=7, stats=partial_groups.PartialGroupStats()
        )
        # A failed slot contributes no reward evidence.
        assert resolved[4].reward == 0.0
        assert await buffer.put(DataBufferInput(prompt_group=group, group=resolved)) is True
        assert len(measured._buffer[0].group) == 8
        return measured._metric_gatherer.collect()

    metrics = asyncio.run(scenario())
    assert seen == [[0, 1, 2, 3, 5, 6, 7]]
    assert metrics["rollout/raw_reward_unfiltered"] == pytest.approx(1 / 7)


def test_dummy_trains_nothing():
    group, failures = failed_group([6])
    resolved = partial_groups.resolve(group, failures, min_size=7, stats=partial_groups.PartialGroupStats())
    data = train_data_conversion.convert_samples_to_train_data(reward_args(), resolved, {}, None, None)
    assert data["loss_masks"][6] == [0] * resolved[6].response_length
    assert data["rewards"][6] == 0.0
    assert len(set(data["rollout_ids"])) == 8


def refresh_producer(min_size, impl):
    producer = RefreshingRolloutFn.__new__(RefreshingRolloutFn)
    producer.args = SimpleNamespace(
        n_samples_per_prompt=8,
        partial_rollout=False,
        group_rm=False,
        sglang_router_policy="cache_aware",
        sglang_enable_deterministic_inference=False,
        olmo_core=SimpleNamespace(refresh_min_group_size=min_size, pipeline_observation_interval=0),
    )
    producer._errors = rollout_errors.RolloutErrors()
    producer._prepared_prompt_ids = {}
    producer._min_group_size = partial_groups.min_group_size(producer.args)
    producer._failed_members = {}
    producer._partial_stats = None
    if producer._min_group_size is not None:
        producer._partial_stats = producer._errors.partial = partial_groups.PartialGroupStats()
    producer._generate_response_impl = impl

    class Admission:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    producer.state = SimpleNamespace(
        args=producer.args,
        aborted=False,
        generate_fn_semaphore=Admission(),
        generate_function=producer._generate_response,
        sampling_params={},
    )
    producer._scheduler = SimpleNamespace(sample_done_callback=None)
    return producer


def prompt_group():
    return [Sample(index=i, group_index=5, prompt=[{"role": "user", "content": "q"}]) for i in range(8)]


def scripted(failures):
    async def impl(input):
        await asyncio.sleep(0)
        if input.sample.index in failures:
            raise failures[input.sample.index]
        complete(input.sample, reward=float(input.sample.index % 2))
        return GenerateFnOutput(samples=input.sample)

    return impl


@pytest.mark.parametrize(
    ("error", "category"),
    [
        (router_failure(), "transport"),
        (httpx.ConnectError("refused"), "transport"),
        (GenerationRequestTimeout("core.refresh_request_timeout"), "timeout"),
    ],
)
def test_producer_keeps_group_after_one_droppable_failure(error, category):
    producer = refresh_producer(7, scripted({3: error}))
    entry = asyncio.run(producer._generate_group(prompt_group()))
    assert len(entry.group) == 8
    assert [PARTIAL_GROUP_PADDING in s.metadata for s in entry.group] == [i == 3 for i in range(8)]
    assert producer._failed_members == {}
    metrics = producer._errors.metrics()
    assert metrics["errors/partial_groups_accepted"] == 1
    assert metrics[f"errors/partial_dropped_{category}_members"] == 1


def test_producer_requeues_below_minimum_with_the_member_error():
    producer = refresh_producer(7, scripted({1: router_failure(), 2: httpx.ConnectError("refused")}))
    with pytest.raises(httpx.HTTPError) as raised:
        asyncio.run(producer._generate_group(prompt_group()))
    assert rollout_errors.recovery_category(raised.value) == "transport"
    assert producer._failed_members == {}
    assert producer._errors.metrics()["errors/partial_requeued_below_min_groups"] == 1


@pytest.mark.parametrize(
    ("min_size", "error"),
    [
        (None, router_failure()),  # default: today's whole-group failure
        (8, router_failure()),
        (7, RuntimeError("Policy refresh request did not finish")),  # not a transport failure
        (7, router_failure("unknown_router_error")),
    ],
)
def test_other_failures_and_default_config_fail_the_whole_group(min_size, error):
    producer = refresh_producer(min_size, scripted({0: error}))
    with pytest.raises(type(error)):
        asyncio.run(producer._generate_group(prompt_group()))
    if min_size in (None, 8):
        assert not any(key.startswith("errors/partial_") for key in producer._errors.metrics())
    assert producer._failed_members == {}


def test_default_config_output_is_unchanged():
    def run(min_size):
        entry = asyncio.run(refresh_producer(min_size, scripted({}))._generate_group(prompt_group()))
        return [(s.index, s.tokens, s.reward, s.metadata, s.loss_mask, s.remove_sample) for s in entry.group]

    assert run(None) == run(7)
    assert not any(key.startswith("errors/partial_") for key in refresh_producer(None, None)._errors.metrics())


@pytest.mark.parametrize("weight", ["router_aux_loss_weight", "router_z_loss_weight"])
def test_partial_groups_guard_router_losses(weight):
    core = SimpleNamespace(refresh_min_group_size=7, **{weight: 0.1})
    args = SimpleNamespace(n_samples_per_prompt=8, olmo_core=core)
    with pytest.raises(ValueError, match=rf"Experimental partial refresh groups require .*{weight}=0"):
        partial_groups.min_group_size(args)
    core.refresh_min_group_size = 8
    assert partial_groups.min_group_size(args) is None


def test_first_padding_row_survives_conversion_and_behavior_metrics():
    original, failures = failed_group([0])
    for sample in original[1:]:
        sample.rollout_routed_experts = np.zeros((len(sample.tokens) - 1, 1, 1), dtype=np.int32)
    group = partial_groups.resolve(original, failures, min_size=7, stats=partial_groups.PartialGroupStats())
    args = reward_args()
    args.use_rollout_routing_replay = True
    batch = train_data_conversion.convert_samples_to_train_data(args, group, {}, None, None)
    batch["total_lengths"] = [len(tokens) for tokens in batch["tokens"]]
    assert batch["rollout_routed_experts"][0].shape == (0, 1, 1)
    assert core_data.policy_versions(batch) == [0] * 7
    assert policy_refresh.validate_batch(batch)[0] == []
    for key in ("tokens", "loss_masks", "rollout_log_probs", "rollout_routed_experts"):
        batch[key] = [torch.as_tensor(row) for row in batch[key]]
    packed = packing.combine(core_data.sample_batches(batch, 64), range(8))
    block = torch.nn.Module()
    block.routed_experts_router = torch.nn.Identity()
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList([block])
    routes = core_data.router_routes(model, packed)["blocks.0.routed_experts_router"]
    assert routes.shape == (1, packed["tokens"].numel(), 1)
    batch["log_probs"] = batch["rollout_log_probs"]
    assert policy_refresh.score_metrics(batch, current_version=0, clip_low=0.8, clip_high=1.2)["response_count"] == 7
    assert observability._active_tokens(batch, current_version=0)[0].numel() == 21
    batch["metadata"][0] = {}
    with pytest.raises(ValueError, match="no provenance"):
        policy_refresh.validate_batch(batch)


@pytest.mark.parametrize("with_padding", [False, True])
def test_rollout_tracking_counts_selected_incomplete_groups_once(monkeypatch, with_padding):
    args = make_args(advantage_estimator="ppo", ci_test=False, log_passrate=False)
    complete_group = [make_sample(0, i) for i in range(4)]
    group = [make_sample(1, i + 4) for i in range(4)]
    if with_padding:
        failures = {id(sample): router_failure() for sample in group[:2]}
        group = partial_groups.resolve(group, failures, min_size=2, stats=partial_groups.PartialGroupStats())
    logged = {}
    monkeypatch.setattr(rollout_metrics.tracking, "log", lambda _args, metrics, **_kwargs: logged.update(metrics))
    rollout_metrics.log_rollout_data(0, args, complete_group + group, {"errors/partial_groups_accepted": 7}, 1.0)
    assert logged["errors/partial_groups_accepted"] == 7
    assert logged["rollout/incomplete_groups"] == int(with_padding)
