# Codex Instructions

Before working in this repository, discover and read all Markdown rule files under `.agents/rules/`, including subdirectories. This repository-root-relative path is a symlink to `.claude/rules/`, the shared source of truth. Treat these files as repository instructions; newly added rules belong to the same set.

When creating or substantially modifying a launcher under `scripts/` or `examples/`, or a model definition under `scripts/models/`, also read and follow `.claude/rules/launch-and-model-scripts.md`.

## Updating from upstream Miles

This repository keeps a filtered upstream lineage, bridged to the existing fork.
This replaces the prior source-only import policy. Preserve the
source exclusions in **every reachable historical tree**, not only at HEAD.
Never attach raw upstream ancestry or restore history before the recorded
baseline. See [the update guide](docs/upstream-updates.md) for the full procedure.

### Upstream provenance

- Repository: https://github.com/radixark/miles
- Original upstream baseline: `e89b45f7f85a0e76fba6a99474b1dd9b67a3c20e`.
- Upstream comparison target: `a22fd824191c64ba7c5d6c531f125406e4ce9cc6`
  (v0.1.2, October 8, 2026).
- Filtered baseline: `19d0f89f4fe6708cca49ac63e28d0115f765a4d2`.
- Filtered target: `d03162fba159188ed16251d66b0e9ce4cbe282d9`.
- Original flattened fork root: `4772a6767f1079eb3548d680c59fe98e481d6ff5`.
- Complete SHA mapping: [v0.1.2-map.json](docs/upstream-history/v0.1.2-map.json).

### Update procedure

1. Fetch raw upstream only into an isolated, unpublished comparison repository.
   Review changes, intermediate versions, renames and copied code against
   `upstream-exclusions.json`, including partial-file removals.
2. Run `tools/filter_upstream_history.py build` into a **new** bare repository,
   using the original baseline, target release and `--previous-map`. The previous
   mapping must reproduce exactly; never silently replace published ancestry.
3. Audit every filtered historical tree, known excluded blob and content move.
   The path/blob checker is not a substitute for reviewing partial copies.
   Resolve uncertain inclusion questions before publishing. Update notices for
   retained source and any newly imported historical adaptations.
4. Fetch only the audited filtered branch into the fork and merge its target.
   The initial bridge is a one-time operation; future updates are normal merges
   from this filtered lineage. Keep Core implementations behind the integration
   interfaces and retain deliberate deletions when resolving conflicts.
5. Run lint, CPU tests and the affected consumer validation gates. Record both
   upstream and filtered SHAs, the mapping, exclusion review and test evidence.
   Publish only the explicit proposal ref, never `--mirror` or `--all`.
6. Merge history-bearing PRs with a **merge commit** (or a true fast-forward).
   Squash/rebase merging discards the ancestry this process is meant to retain.

### Integration and exclusion policy

Keep backend-specific implementations behind the small interfaces in
`miles/backends/hooks.py`, with Core implementations under
`miles/backends/core_utils/`. Put fork-added regression tests in separate files
instead of appending them to upstream test modules.

Use `upstream-exclusions.json` when resolving deliberately deleted paths. Preserve
those deletions and excluded prefixes, and review relocated paths at their new
locations. Run `python tools/check_upstream_exclusions.py` before committing an
update. This check cannot identify renamed/copied implementations or partial-file
removals; follow [the update guide](docs/upstream-updates.md) for that review.
