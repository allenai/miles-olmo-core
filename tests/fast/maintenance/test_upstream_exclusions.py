import json
from pathlib import Path

from tools.check_upstream_exclusions import excluded_paths

ROOT = Path(__file__).resolve().parents[3]
POLICY = json.loads((ROOT / "upstream-exclusions.json").read_text())


def test_old_paths_and_new_component_files_remain_excluded():
    paths = [
        "miles/utils/nvfp4_fake_qat.py",
        "examples/new_example/train.py",
        "miles_plugins/models/deepseek_v4/new_kernel.py",
        "miles/backends/core_utils/integration.py",
        "miles/utils/arguments.py",
    ]
    assert excluded_paths(paths, POLICY) == sorted(paths[:3])


def test_relocation_preserves_destination_without_resurrecting_source():
    for source, destination in POLICY["relocated_paths"].items():
        assert excluded_paths([source, destination], POLICY) == [source]
        assert (ROOT / destination).is_file()


def test_manifest_covers_removed_files_but_not_itself():
    assert excluded_paths(["upstream-exclusions.json"], POLICY) == []
    assert not any((ROOT / path).exists() for path in POLICY["excluded_paths"])
    assert not any((ROOT / prefix).exists() for prefix in POLICY["excluded_prefixes"])
