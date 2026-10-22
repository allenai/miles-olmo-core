"""Exercise real Git ancestry, exclusion and repeatability contracts offline."""

import json
import subprocess
from pathlib import Path

import pytest

from tools.filter_upstream_history import TYPER, audit, build, git, policy_digest, review_content, strip_stream_adapter


@pytest.fixture
def history(tmp_path, monkeypatch):
    for key, value in {
        "GIT_AUTHOR_NAME": "Upstream Author",
        "GIT_AUTHOR_EMAIL": "author@example.invalid",
        "GIT_COMMITTER_NAME": "Upstream Committer",
        "GIT_COMMITTER_EMAIL": "committer@example.invalid",
        "GIT_AUTHOR_DATE": "1700000000 +0000",
        "GIT_COMMITTER_DATE": "1700000001 +0000",
    }.items():
        monkeypatch.setenv(key, value)
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init", "-b", "main")

    def commit(path, content, message):
        file = source / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content)
        git(source, "add", ".")
        git(source, "commit", "-m", message)
        return git(source, "rev-parse", "HEAD").decode().strip()

    commit("kept.py", "answer = 1\n", "Before baseline")
    baseline = commit("excluded/secret.py", "excluded_implementation = 1\n", "Baseline")
    changed = commit("kept.py", "answer = 2\n", "Useful change")
    empty = commit("excluded/secret.py", "excluded_implementation = 2\n", "Excluded-only change")
    policy = {"excluded_paths": [], "excluded_prefixes": ["excluded/"], "relocated_paths": {}}
    return source, baseline, changed, empty, policy, commit


def test_reproducible_lineage_preserves_empty_commits_and_metadata(history, tmp_path):
    source, baseline, changed, target, policy, _ = history
    first = tmp_path / "first.git"
    result = build(source, first, baseline, target, policy)
    assert result == build(source, tmp_path / "second.git", baseline, target, policy)
    mapped = result["commits"]
    assert len(mapped) == 3
    assert result["empty_commits"] == 1
    assert git(first, "rev-list", "--parents", mapped[baseline]).decode().strip() == mapped[baseline]
    assert git(first, "rev-parse", f"{mapped[target]}^").decode().strip() == mapped[changed]
    assert git(first, "rev-parse", f"{mapped[changed]}^{{tree}}") == git(
        first, "rev-parse", f"{mapped[target]}^{{tree}}"
    )
    for original, filtered in mapped.items():
        assert git(source, "show", "-s", "--format=%an|%ae|%at|%cn|%ce|%ct|%s", original) == git(
            first, "show", "-s", "--format=%an|%ae|%at|%cn|%ce|%ct|%s", filtered
        )
        with pytest.raises(subprocess.CalledProcessError):
            git(first, "cat-file", "-e", original)
    for oid in result["blocked_blob_ids"]:
        with pytest.raises(subprocess.CalledProcessError):
            git(first, "cat-file", "-e", oid)
    assert audit(first, mapped[target], policy, result)["reachable_commits"] == 3


def test_next_update_extends_published_mapping(history, tmp_path):
    source, baseline, _, target, policy, commit = history
    previous = build(source, tmp_path / "old.git", baseline, target, policy)
    next_target = commit("new.py", "new = True\n", "Next release")
    updated = build(source, tmp_path / "new.git", baseline, next_target, policy, previous)
    for original, filtered in previous["commits"].items():
        assert updated["commits"][original] == filtered
    assert (
        git(tmp_path / "new.git", "rev-parse", f'{updated["filtered_target"]}^').decode().strip()
        == previous["filtered_target"]
    )


def test_changed_historical_policy_requires_explicit_migration(history, tmp_path):
    source, baseline, _, target, policy, _ = history
    previous = build(source, tmp_path / "old.git", baseline, target, policy)
    changed_policy = {**policy, "excluded_paths": ["kept.py"]}
    with pytest.raises(ValueError, match="Published mapping changed"):
        build(source, tmp_path / "changed.git", baseline, target, changed_policy, previous)


def test_preserves_merge_parents(history, tmp_path):
    source, baseline, _, target, policy, commit = history
    git(source, "checkout", "-b", "side", baseline)
    side = commit("side.py", "side = True\n", "Side branch")
    git(source, "checkout", "main")
    git(source, "merge", "--no-ff", "side", "-m", "Merge side")
    merged = git(source, "rev-parse", "HEAD").decode().strip()
    output = tmp_path / "merged.git"
    mapping = build(source, output, baseline, merged, policy)["commits"]
    parents = git(output, "show", "-s", "--format=%P", mapping[merged]).decode().split()
    assert parents == [mapping[target], mapping[side]]


def test_exact_copy_of_excluded_blob_at_new_path_is_rejected(history, tmp_path):
    source, baseline, _, _, policy, commit = history
    target = commit("renamed.py", "excluded_implementation = 2\n", "Move outside exclusion")
    with pytest.raises(ValueError, match="Excluded source blobs remain reachable"):
        build(source, tmp_path / "bad.git", baseline, target, policy)


def test_audit_rejects_exclusion_in_intermediate_commit_even_if_head_is_clean(history, tmp_path):
    source, baseline, _, target, policy, _ = history
    output = tmp_path / "filtered.git"
    result = build(source, output, baseline, target, policy)
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "clone", "-b", "filtered-upstream", str(output), str(checkout)], check=True)
    file = checkout / "excluded/leak.py"
    file.parent.mkdir()
    file.write_text("historical_leak = True\n")
    git(checkout, "add", ".")
    git(checkout, "commit", "-m", "Accidental restoration")
    git(checkout, "rm", "excluded/leak.py")
    git(checkout, "commit", "-m", "Clean HEAD does not clean history")
    with pytest.raises(ValueError, match="Excluded paths in reachable tree"):
        audit(checkout, "HEAD", policy, result)


def test_content_review_finds_modified_copy_outside_excluded_path(history, tmp_path):
    source, baseline, _, _, policy, commit = history
    excluded = "\n".join(f"excluded_variable_{i} = {i}" for i in range(15)) + "\n"
    commit("excluded/secret.py", excluded, "Excluded implementation")
    target = commit("copy.py", "# A new name and comment do not remove copied content\n" + excluded, "Modified copy")
    output = tmp_path / "copy.git"
    result = build(source, output, baseline, target, policy)
    report = review_content(source, output, policy, result)
    match = next(row for row in report["matches"] if row["paths"] == ["copy.py"])
    assert match["overlaps"]["excluded/secret.py"] == 6
    assert "excluded_variable_" not in json.dumps(report)


def test_side_branch_from_before_baseline_requires_boundary_review(history, tmp_path):
    source, baseline, _, target, policy, commit = history
    git(source, "checkout", "-b", "old-side", f"{baseline}^")
    commit("side.py", "old_side = True\n", "Old side")
    git(source, "checkout", "main")
    git(source, "merge", "--no-ff", "old-side", "-m", "Merge old side")
    target = git(source, "rev-parse", "HEAD").decode().strip()
    with pytest.raises(ValueError, match="predates the baseline"):
        build(source, tmp_path / "old-side.git", baseline, target, policy)


@pytest.mark.parametrize("condition", ["client_stream", "response_intent.stream"])
def test_stream_filter_retains_nonstream_code_and_refuses_surprises(condition):
    original = (
        "def _response_to_stream_chunk(response):\n"
        "    return {'excluded': response}\n\n"
        "def _chat_client_response(result, response, client_stream):\n"
        f"    if {condition}:\n"
        "        return _response_to_stream_chunk(response)\n"
        "    return response\n"
    ).encode()
    filtered = strip_stream_adapter(original)
    assert b"excluded':" not in filtered
    assert b"_response_to_stream_chunk" not in filtered
    assert filtered.endswith(b"    return response\n")
    with pytest.raises(ValueError, match="Unreviewed"):
        strip_stream_adapter(original.replace(condition.encode(), b"new_stream_flag"))


def test_partial_removal_typer_and_approved_relocation(history, tmp_path):
    source, baseline, _, _, policy, commit = history
    commit(TYPER, "unreviewed_typer_implementation = True\n", "Old typer")
    target = commit("excluded/approved.py", "approved_helper = True\n", "Approved helper")
    policy = {**policy, "relocated_paths": {"excluded/approved.py": "helpers/approved.py"}}
    output = tmp_path / "relocated.git"
    result = build(source, output, baseline, target, policy)
    paths = git(output, "ls-tree", "-r", "--name-only", result["filtered_target"]).decode().splitlines()
    assert "helpers/approved.py" in paths
    assert TYPER not in paths
    assert not any(path.startswith("excluded/") for path in paths)


def test_real_mapping_is_complete_and_uses_current_policy():
    root = Path(__file__).resolve().parents[3]
    mapping = json.loads((root / "docs/upstream-history/v0.1.2-map.json").read_text())
    assert len(mapping["commits"]) == 451
    assert mapping["policy_sha256"] == policy_digest(json.loads((root / "upstream-exclusions.json").read_text()))
    assert mapping["commits"][mapping["upstream_baseline"]] == mapping["filtered_baseline"]
    assert mapping["commits"][mapping["upstream_target"]] == mapping["filtered_target"]
