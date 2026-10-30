"""Regressions for false misses observed in the October Core RL cache audit."""

import copy

import pytest
from tests.fast.compiler_cache import test_compiler_cache as helpers

from miles.utils.compiler_cache import compiler_cache as cache


def core_identity():
    args = helpers.identity()
    args["scope"] = cache.CORE_TRITON_SCOPE
    args["run_config"].update(worker_slot="inference-engine-all-0-0-0-0")
    args["run_config"]["core"]["max_policy_lag"] = 0
    args["run_config"]["miles"].update(
        rollout_num_gpus_per_engine=1,
        rollout_num_gpus=4,
        global_batch_size=32,
        n_samples_per_prompt=4,
        rollout_batch_size=8,
        sglang_server_concurrency=64,
    )
    args["toolchain"]["torch_hardware"] = {
        "gpus": [{"name": "NVIDIA B300", "major": 10, "minor": 3, "memory": 287431131136, "sm_count": 148}] * 8
    }
    return args


def assert_reuses_published(tmp_path, before, after):
    original = copy.deepcopy(before)
    key = cache.fingerprint(**before)[0]
    source = helpers.private(tmp_path, "producer")
    helpers.write(source / "triton", "kernel.so")
    cache.publish(tmp_path / "shared", source, key, "triton")
    new_key = cache.fingerprint(**after)[0]
    assert new_key == key
    target = helpers.private(tmp_path, "consumer")
    assert cache.restore(tmp_path / "shared", target, new_key, "triton")["status"] == "hit"
    assert (target / "triton/kernel.so").read_bytes() == b"compiled"
    assert before == original


@pytest.mark.parametrize("slot", ["trainer-engine-actor-0-0", "inference-engine-all-0-0-0-0"])
def test_audit_a_to_b_reuses_cache(tmp_path, slot):
    before = core_identity()
    before["run_config"]["worker_slot"] = slot
    after = copy.deepcopy(before)
    after["run_config"]["core"]["max_policy_lag"] = 6
    after["run_config"]["miles"].update(
        global_batch_size=128, n_samples_per_prompt=8, rollout_batch_size=16, sglang_server_concurrency=128
    )
    assert_reuses_published(tmp_path, before, after)


@pytest.mark.parametrize("index,visible", [(1, 8), (12, 7), (18, 7)])
def test_equivalent_serving_workers_reuse_cache(tmp_path, index, visible):
    before = core_identity()
    after = copy.deepcopy(before)
    after["run_config"]["worker_slot"] = f"inference-engine-all-0-0-{index}-0"
    after["run_config"]["miles"].update(rollout_num_gpus=19, async_max_concurrent_samples=4864)
    after["toolchain"]["torch_hardware"]["gpus"] = after["toolchain"]["torch_hardware"]["gpus"][:visible]
    assert_reuses_published(tmp_path, before, after)


@pytest.mark.parametrize("slot,tp", [("trainer-engine-actor-0-0", 1), ("inference-engine-all-0-0-0-0", 2)])
def test_trainer_ranks_and_multigpu_serving_remain_separate(slot, tp):
    args = core_identity()
    args["run_config"]["worker_slot"] = slot
    args["run_config"]["miles"]["rollout_num_gpus_per_engine"] = tp
    key = cache.fingerprint(**args)[0]
    args["run_config"]["worker_slot"] = slot[:-1] + "1"
    assert cache.fingerprint(**args)[0] != key


@pytest.mark.parametrize("field,value", [("major", 9), ("memory", 85029158912), ("sm_count", 132)])
def test_different_or_mixed_hardware_is_not_collapsed(field, value):
    args = core_identity()
    baseline = cache.fingerprint(**args)[0]
    args["toolchain"]["torch_hardware"]["gpus"] = copy.deepcopy(args["toolchain"]["torch_hardware"]["gpus"][:1])
    args["toolchain"]["torch_hardware"]["gpus"][0][field] = value
    assert cache.fingerprint(**args)[0] != baseline
    args["toolchain"]["torch_hardware"]["gpus"].append(core_identity()["toolchain"]["torch_hardware"]["gpus"][0])
    assert cache.fingerprint(**args)[0] != baseline
    mixed = cache.fingerprint(**args)[0]
    args["run_config"]["worker_slot"] = "inference-engine-all-0-0-1-0"
    assert cache.fingerprint(**args)[0] != mixed


def test_unrelated_megatron_lock_changes_reuse_core_cache(tmp_path):
    before = core_identity()
    before["runtime_lock"]["sources"] = {"miles": {"revision": "core-runtime"}}
    after = copy.deepcopy(before)
    after["runtime_lock"]["sources"].update(
        {
            name: {"repository": "different-repository", "revision": "different"}
            for name in ("megatron-lm", "megatron-bridge", "olmo-megatron")
        }
    )
    after["runtime_lock"]["source_patches"] = [
        {"source": "olmo-megatron", "target": "megatron-lm", "files": ["a.patch"]}
    ]
    assert_reuses_published(tmp_path, before, after)
    after["runtime_lock"]["source_patches"][0]["target"] = "olmo-core"
    assert cache.fingerprint(**after)[0] != cache.fingerprint(**before)[0]
    patched = cache.fingerprint(**after)[0]
    after["runtime_lock"]["sources"]["olmo-megatron"]["revision"] = "new-active-patch-source"
    assert cache.fingerprint(**after)[0] != patched
    after = copy.deepcopy(before)
    after["runtime_lock"]["sources"]["future-backend"] = {"revision": "new"}
    assert cache.fingerprint(**after)[0] != cache.fingerprint(**before)[0]


def test_application_eval_opd_and_tests_do_not_invalidate(tmp_path):
    app = tmp_path / "open_instruct"
    active = helpers.write(app, "miles/execution/driver.py", b"kernel_control = 1\n")
    excluded = [
        helpers.write(app, path, b"unrelated = 1\n")
        for path in ("miles/evaluation/runner.py", "miles/distillation/loss.py", "test_eval.py", "tests/test_train.py")
    ]
    before = core_identity()
    before["sources"]["open-instruct"] = cache.core_application_source(app)
    strict_source = cache.source_identity(app)
    for path in excluded:
        path.write_text("unrelated = 2\n")
    after = copy.deepcopy(before)
    after["sources"]["open-instruct"] = cache.core_application_source(app)
    assert_reuses_published(tmp_path, before, after)
    assert cache.source_identity(app) != strict_source
    active.write_text("kernel_control = 2\n")
    after["sources"]["open-instruct"] = cache.core_application_source(app)
    assert cache.fingerprint(**before)[0] != cache.fingerprint(**after)[0]
    helpers.write(app, "new_runtime_module.py", b"unknown = True\n")
    assert cache.core_application_source(app) != after["sources"]["open-instruct"]


def test_scope_is_explicit_and_does_not_alias_legacy_or_experimental_cache():
    args = core_identity()
    normalized = cache.fingerprint(**args)[0]
    args["scope"] = "strict"
    strict = cache.fingerprint(**args)[0]
    assert strict != normalized
    args["run_config"]["miles"]["sglang_server_concurrency"] *= 2
    assert cache.fingerprint(**args)[0] != strict
    args["scope"] = "unknown"
    with pytest.raises(ValueError, match="compatibility scope"):
        cache.fingerprint(**args)
