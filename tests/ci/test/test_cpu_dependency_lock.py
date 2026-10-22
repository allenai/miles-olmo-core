"""Exercise the hosted workflow's actual dependency resolution and overrides."""

import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("override", ["none", "release", "body", "dispatch"])
def test_cpu_sources_are_pinned_and_explicit_overrides_win(tmp_path, override):
    lock_path = Path("tests/ci/cpu-dependency-lock.json")
    lock = json.loads((ROOT / lock_path).read_text())
    assert all(re.fullmatch(r"[0-9a-f]{40}", value) for value in lock.values())
    (tmp_path / lock_path).parent.mkdir(parents=True)
    (tmp_path / lock_path).write_text(json.dumps(lock))
    expected = {"ci_sglang_pr": lock["sglang_commit"], "ci_megatron_pr": lock["megatron_commit"]}
    env = os.environ | {
        "PR_BODY": "",
        "INPUT_SGLANG_PR": "",
        "INPUT_MEGATRON_PR": "",
        "GITHUB_OUTPUT": str(tmp_path / "output"),
    }
    if override != "none":
        (tmp_path / "release-lock.json").write_text(
            json.dumps({"sglang_commit": "a" * 40, "megatron_commit": "b" * 40})
        )
        expected = {"ci_sglang_pr": "a" * 40, "ci_megatron_pr": "b" * 40}
    if override in ("body", "dispatch"):
        env["PR_BODY"] = "ci-sglang-pr: #123\nci-megatron-pr: test-megatron\n"
        expected = {"ci_sglang_pr": "refs/pull/123/head", "ci_megatron_pr": "test-megatron"}
    if override == "dispatch":
        env.update(INPUT_SGLANG_PR="test-sglang", INPUT_MEGATRON_PR="#456")
        expected = {"ci_sglang_pr": "test-sglang", "ci_megatron_pr": "refs/pull/456/head"}
    workflow = yaml.safe_load((ROOT / ".github/workflows/_run-cpu-ci.yml").read_text())
    step = next(step for step in workflow["jobs"]["run-cpu"]["steps"] if step.get("id") == "resolve-refs")
    subprocess.run(["bash", "-e", "-o", "pipefail", "-c", step["run"]], cwd=tmp_path, env=env, check=True)
    actual = dict(line.split("=", 1) for line in (tmp_path / "output").read_text().splitlines())
    assert actual == expected
