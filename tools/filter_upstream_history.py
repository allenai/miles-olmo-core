"""Build a deterministic, isolated upstream lineage with source exclusions.

Only filtered blobs are written to the new bare repository. The initial upstream
baseline becomes a root snapshot; subsequent commits retain their topology,
authors, dates and messages, including commits made empty by the filter. Original
signatures cannot survive rewriting. This tool does not push or merge anything.
"""

import argparse
import ast
import hashlib
import json
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

FORMAT_VERSION = 1
SESSION_CORE = "miles/rollout/session/core.py"
TYPER = "miles/utils/typer_utils.py"


def git(repo, *args, data=None, env=None):
    return subprocess.check_output(["git", "-C", str(repo), *args], input=data, env=env)


def policy_digest(policy):
    rules = {key: policy[key] for key in ("excluded_paths", "excluded_prefixes", "relocated_paths")}
    return hashlib.sha256(json.dumps(rules, sort_keys=True).encode()).hexdigest()


def destination(path, policy):
    # These two reviewed runtime helpers are exceptions to the examples ban.
    if path in policy["relocated_paths"]:
        return policy["relocated_paths"][path]
    if path == TYPER or path in policy["excluded_paths"] or path.startswith(tuple(policy["excluded_prefixes"])):
        return None
    return path


def strip_stream_adapter(data):
    """Remove the excluded helper and its SSE branch, preserving other code.

    Accept only the two reviewed upstream call shapes. A future refactor must
    be reviewed instead of silently retaining or over-deleting implementation.
    The filtered lineage is for merging; only fork merge heads are runnable.
    """
    text = data.decode("utf-8")
    nodes = {node.name: node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)}
    helper = nodes.get("_response_to_stream_chunk")
    caller = nodes.get("_chat_client_response")
    if helper is None or caller is None or helper.decorator_list:
        raise ValueError("Unreviewed session streaming layout")
    branch = caller.body[0]
    if not isinstance(branch, ast.If) or branch.orelse:
        raise ValueError("Unreviewed session streaming branch")
    if ast.unparse(branch.test) not in {"client_stream", "response_intent.stream"}:
        raise ValueError("Unreviewed session streaming condition")
    lines = text.splitlines(keepends=True)
    replacements = [
        (helper.lineno - 1, helper.end_lineno, []),
        (
            branch.lineno - 1,
            branch.end_lineno,
            [lines[branch.lineno - 1], '        raise ValueError("Streaming is excluded from this fork")\n'],
        ),
    ]
    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start:end] = replacement
    result = "".join(lines)
    if "_response_to_stream_chunk" in result or "ProRL-Agent-Server" in result:
        raise ValueError("Streaming adaptation remains outside the reviewed removal")
    ast.parse(result)
    return result.encode()


def tree_entries(repo, commit):
    result = {}
    for row in git(repo, "ls-tree", "-rz", commit).split(b"\0"):
        if not row:
            continue
        metadata, path = row.split(b"\t", 1)
        mode, kind, oid = metadata.decode().split()
        if kind != "blob":
            raise ValueError(f"Review non-blob entry before importing: {path!r}")
        result[path.decode()] = (mode, oid)
    return result


def write_object(repo, kind, data):
    return git(repo, "hash-object", "-w", "-t", kind, "--stdin", data=data).decode().strip()


def rewritten_commit(raw, tree, parents, original):
    header, message = raw.split(b"\n\n", 1)
    # Only these metadata headers remain valid after a rewrite. In particular,
    # gpgsig and mergetag would falsely attest to the original object.
    metadata = [line for line in header.splitlines() if line.startswith((b"author ", b"committer ", b"encoding "))]
    if sum(line.startswith(b"author ") for line in metadata) != 1:
        raise ValueError("Missing or duplicate commit author")
    if sum(line.startswith(b"committer ") for line in metadata) != 1:
        raise ValueError("Missing or duplicate commit committer")
    lines = [f"tree {tree}".encode(), *(f"parent {parent}".encode() for parent in parents), *metadata]
    trailer = f"\n\nOriginal-Upstream-Commit: {original}\nUpstream-History-Filter: v{FORMAT_VERSION}\n".encode()
    return b"\n".join(lines) + b"\n\n" + message.rstrip(b"\n") + trailer


def commit_order(source, baseline, target):
    git(source, "merge-base", "--is-ancestor", baseline, target)
    rows = git(source, "rev-list", "--reverse", "--topo-order", "--parents", f"{baseline}..{target}")
    order = [(baseline, [])]
    seen = {baseline}
    for row in rows.decode().splitlines():
        original, *parents = row.split()
        if not set(parents) <= seen:
            raise ValueError("A merged side branch predates the baseline; review its boundary explicitly")
        order.append((original, parents))
        seen.add(original)
    return order


def filtered_tree(source, output, original, policy, cache, blocked):
    result = {}
    for path, (mode, oid) in tree_entries(source, original).items():
        dest = destination(path, policy)
        if dest is None:
            blocked.add(oid)
            continue
        key = (oid, path == SESSION_CORE)
        if key not in cache:
            data = git(source, "cat-file", "blob", oid)
            if path == SESSION_CORE:
                blocked.add(oid)
                data = strip_stream_adapter(data)
            cache[key] = write_object(output, "blob", data)
        entry = (mode, cache[key])
        if dest in result and result[dest] != entry:
            raise ValueError(f"Conflicting relocation destination: {dest}")
        result[dest] = entry
    return result


def write_tree(output, entries, previous):
    changes = []
    for path in sorted(set(entries) | set(previous)):
        if entries.get(path) == previous.get(path):
            continue
        mode, oid = entries.get(path, ("0", "0" * 40))
        changes.append(f"{mode} {oid}\t{path}".encode() + b"\0")
    git(output, "update-index", "-z", "--index-info", data=b"".join(changes))
    return git(output, "write-tree").decode().strip()


def verify_previous(previous, mapping, source, output):
    if previous is None:
        return
    if previous["format_version"] != FORMAT_VERSION:
        raise ValueError("Filter format changed; an explicit history migration is required")
    for original, old_filtered in previous["commits"].items():
        if mapping.get(original) != old_filtered:
            raise ValueError(f"Published mapping changed at {original}; refusing a new lineage")
    # The complete previous prefix must survive, not merely its release tree.
    git(source, "merge-base", "--is-ancestor", previous["upstream_target"], next(reversed(mapping)))
    git(output, "merge-base", "--is-ancestor", previous["filtered_target"], next(reversed(mapping.values())))


def build(source, output, baseline, target, policy, previous=None):
    if output.exists():
        raise ValueError("Output must be a new directory; never rewrite a published checkout")
    baseline = git(source, "rev-parse", f"{baseline}^{{commit}}").decode().strip()
    target = git(source, "rev-parse", f"{target}^{{commit}}").decode().strip()
    order = commit_order(source, baseline, target)
    subprocess.run(["git", "init", "--bare", "--object-format=sha1", str(output)], check=True, capture_output=True)
    mapping, cache, blocked, prior_entries = {}, {}, set(), {}
    empty = 0
    for original, parents in order:
        entries = filtered_tree(source, output, original, policy, cache, blocked)
        tree = write_tree(output, entries, prior_entries)
        mapped_parents = [mapping[parent] for parent in parents]
        if mapped_parents and tree == git(output, "rev-parse", f"{mapped_parents[0]}^{{tree}}").decode().strip():
            empty += 1
        raw = git(source, "cat-file", "commit", original)
        mapping[original] = write_object(output, "commit", rewritten_commit(raw, tree, mapped_parents, original))
        prior_entries = entries
    verify_previous(previous, mapping, source, output)
    git(output, "update-ref", "refs/heads/filtered-upstream", mapping[target])
    result = {
        "format_version": FORMAT_VERSION,
        "upstream_baseline": baseline,
        "upstream_target": target,
        "filtered_baseline": mapping[baseline],
        "filtered_target": mapping[target],
        "policy_sha256": policy_digest(policy),
        "commits": mapping,
        "empty_commits": empty,
        "blocked_blob_ids": sorted(blocked),
    }
    audit(output, mapping[target], policy, result)
    return result


def audit(repo, head, policy, mapping):
    """Check every reachable tree and known prohibited blob, not only HEAD.

    Exact blocked blobs that also occur in retained paths (e.g. empty __init__
    files) are reported for review rather than classified as implementation.
    Content similarity/rename review remains a separate publication gate.
    """
    forbidden = set(policy["excluded_paths"]) | set(policy["relocated_paths"])
    prefixes = tuple(policy["excluded_prefixes"])
    trees = set(git(repo, "log", "--format=%T", head).decode().splitlines())
    objects = set()
    for tree in trees:
        entries = tree_entries(repo, tree)
        bad = [path for path in entries if path in forbidden or path.startswith(prefixes)]
        if bad:
            raise ValueError(f"Excluded paths in reachable tree {tree}: {bad}")
        objects.update(oid for _, oid in entries.values())
    shared = sorted(objects & set(mapping["blocked_blob_ids"]))
    # An empty package marker has no excluded implementation. All other shared
    # blobs require an explicit review; do not silently whitelist them by size.
    shared = [oid for oid in shared if git(repo, "cat-file", "blob", oid).strip()]
    if shared:
        raise ValueError(f"Excluded source blobs remain reachable: {shared}")
    originals = set(mapping["commits"])
    reachable = set(git(repo, "rev-list", head).decode().splitlines())
    if originals & reachable:
        raise ValueError("Unfiltered original commits are reachable")
    if not set(mapping["commits"].values()) <= reachable:
        raise ValueError("Filtered upstream commits are missing from the proposed head")
    return {"reachable_commits": len(reachable), "unique_trees": len(trees), "unique_blobs": len(objects)}


def content_windows(data):
    try:
        text = data.decode()
    except UnicodeDecodeError:
        return set()
    lines = [
        "".join(line.split())
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "//"))
    ]
    return {hashlib.sha256("\n".join(lines[i : i + 10]).encode()).digest() for i in range(len(lines) - 9)}


def review_content(source, repo, policy, mapping):
    """Generate similarity leads, never an automatic clearance decision.

    Compare ten normalized nonblank/noncomment lines across every version of
    excluded source and every retained historical blob, plus commit messages.
    Report paths/hashes/counts only: do not publish the excluded source snippets.
    """
    excluded, retained = defaultdict(set), defaultdict(set)
    for original, filtered in mapping["commits"].items():
        for path, (_, oid) in tree_entries(source, original).items():
            if destination(path, policy) is None or path == SESSION_CORE:
                excluded[oid].add(path)
        for path, (_, oid) in tree_entries(repo, filtered).items():
            retained[oid].add(path)
    index = defaultdict(set)
    for oid, paths in excluded.items():
        data = git(source, "cat-file", "blob", oid)
        if paths == {SESSION_CORE}:
            # Only the removed adaptation is excluded, not the retained core.
            text = data.decode()
            lines = text.splitlines(keepends=True)
            nodes = {node.name: node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)}
            removed = [nodes["_response_to_stream_chunk"], nodes["_chat_client_response"].body[0]]
            data = "".join("".join(lines[node.lineno - 1 : node.end_lineno]) for node in removed).encode()
        for window in content_windows(data):
            index[window].update(paths)
    matches = []
    for kind, objects in (("blob", retained), ("commit", {oid: {oid} for oid in mapping["commits"].values()})):
        for oid, paths in objects.items():
            data = git(repo, "cat-file", kind, oid)
            if kind == "commit":
                data = data.split(b"\n\n", 1)[1]
            overlaps = Counter(path for window in content_windows(data) for path in index.get(window, ()))
            if overlaps:
                matches.append({"kind": kind, "object": oid, "paths": sorted(paths), "overlaps": dict(overlaps)})
    return {
        "excluded_blob_versions": len(excluded),
        "retained_blob_versions": len(retained),
        "matches": matches,
        "limitation": "Heuristic review leads; renamed/extracted/rewritten copies still require inspection.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("build")
    create.add_argument("--source", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--baseline", required=True)
    create.add_argument("--target", required=True)
    create.add_argument("--previous-map", type=Path)
    check = commands.add_parser("audit")
    check.add_argument("--repo", type=Path, required=True)
    check.add_argument("--head", default="HEAD")
    scan = commands.add_parser("scan")
    scan.add_argument("--source", type=Path, required=True)
    scan.add_argument("--repo", type=Path, required=True)
    scan.add_argument("--report", type=Path, required=True)
    for command in (create, check, scan):
        command.add_argument("--policy", type=Path, default=Path("upstream-exclusions.json"))
        command.add_argument("--map", type=Path, required=True)
    args = parser.parse_args()
    # Never inherit an external index/object store, replacement refs or grafts.
    for key in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    ):
        if key in os.environ:
            parser.error(f"Unset {key} before filtering")
    os.environ["GIT_NO_REPLACE_OBJECTS"] = "1"
    policy = json.loads(args.policy.read_text())
    if args.command == "build":
        previous = json.loads(args.previous_map.read_text()) if args.previous_map else None
        result = build(args.source, args.output, args.baseline, args.target, policy, previous)
        args.map.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key not in {"commits", "blocked_blob_ids"}}))
    elif args.command == "audit":
        print(json.dumps(audit(args.repo, args.head, policy, json.loads(args.map.read_text()))))
    else:
        result = review_content(args.source, args.repo, policy, json.loads(args.map.read_text()))
        args.report.write_text(json.dumps(result, indent=2) + "\n")
        print(f"{len(result['matches'])} similarity leads written to {args.report}; manual review required")


if __name__ == "__main__":
    main()
