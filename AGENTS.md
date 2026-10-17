# Codex Instructions

When creating or substantially modifying `miles/**/*.py`, `scripts/**/*.py`, `tools/**/*.py`, `train.py`, or `train_async.py`, read and follow `.claude/rules/general-code-style.md`.

When creating or substantially modifying a launcher under `scripts/` or `examples/`, or a model definition under `scripts/models/`, also read and follow `.claude/rules/launch-and-model-scripts.md`.

## Updating from upstream Miles

This repository's history was flattened to avoid distributing historical code
whose inclusion was legally uncertain. Preserve that boundary when updating from
upstream; do not restore the pre-flattening history or attach upstream ancestry.

### Upstream provenance

- Repository: https://github.com/radixark/miles
- Upstream comparison base: `e89b45f7f85a0e76fba6a99474b1dd9b67a3c20e`
  (September 22, 2026, `fix(train): hand off to the engine before an off-cadence final eval (#3615)`).
- Flattened release root: `4772a6767f1079eb3548d680c59fe98e481d6ff5`.

The upstream SHA records provenance as text; it is not a parent of the flattened
release. The release includes fork-specific changes and intentional source
removals, so it is not an exact copy of that upstream commit.

### Update procedure

1. Prepare the update in a separate local checkout with upstream history
   available. Use the recorded upstream comparison base, the fork's current
   source tree, and the chosen upstream target for a three-way comparison.
2. Review the resulting source changes and resolve conflicts while preserving
   fork-specific behavior and intentional exclusions. Check upstream additions
   and modifications for reintroduced components from the source cleanup;
   conflict-free application alone is not sufficient. Document the exclusions
   retained or changed and keep `THIRD_PARTY_NOTICES.md` consistent with the
   resulting distributed source. Escalate unresolved inclusion questions before
   bringing the affected code into this repository.
3. Bring only the reviewed source changes into this repository and run the
   checks appropriate to those changes. Commit them as ordinary single-parent
   commits descended from the flattened release. Do not pull/merge upstream
   directly, use `--allow-unrelated-histories`, or publish branches or tags that
   reconnect or expose the old history.
4. Record the upstream target SHA and update scope in the commit message. After
   a complete reviewed update, advance the comparison base above in the same
   commit. If importing only selected fixes, record their SHAs separately and
   leave the comparison base unchanged so later updates do not skip changes.

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
