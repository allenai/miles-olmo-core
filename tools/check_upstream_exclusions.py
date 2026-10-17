"""Report excluded source paths without deleting or resolving anything.

Run against a candidate checkout, or pass a newline-delimited upstream changed
path list with --paths-from. A clean result does not detect renamed/copied code;
see docs/upstream-updates.md for the required content review.
"""

import argparse
import json
import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def excluded_paths(paths, policy) -> list[str]:
    exact = set(policy["excluded_paths"]) | set(policy["relocated_paths"])
    prefixes = tuple(policy["excluded_prefixes"])
    return sorted({path for path in paths if path in exact or path.startswith(prefixes)})


def checkout_paths(root: Path) -> list[str]:
    output = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"], text=True
    )
    return [path for path in output.split("\0") if path and (root / path).is_file()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=REPOSITORY_ROOT / "upstream-exclusions.json")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--tree", type=Path, default=REPOSITORY_ROOT)
    source.add_argument("--paths-from", type=Path)
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text())
    paths = args.paths_from.read_text().splitlines() if args.paths_from else checkout_paths(args.tree)
    blocked = excluded_paths(paths, policy)
    if blocked:
        print("Preserve exclusion; review before importing:")
        print("\n".join(blocked))
        return 1
    print("No excluded paths found. Renames, copied code, and partial-file removals still require review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
