# Maintaining the OLMo-core fork

The comparison base and source-only update procedure live in [AGENTS.md](../AGENTS.md).
The fork's flattened ancestry is intentional: prepare comparisons in a separate
upstream checkout and import only reviewed source. Do not merge upstream Git
history into this repository. Selected backports do not advance the comparison
base; a complete reviewed update does.

## Keep the integration seams small

- `miles/backends/hooks.py` defines three synchronous/asynchronous boundaries:
  rollout path selection, producer shutdown, and checkpoint finalization.
  Native backends have no-op defaults. Errors propagate to the caller.
- `miles/backends/core_utils/integration.py` implements Core policy. Its argument
  loader imports Open Instruct only when Core arguments are requested.
- `RolloutExecutor.core_publication_boundary` and `core_engine_drain` remain public
  RPC methods. They delegate to Core helpers. Producer mutations still execute
  on the producer event loop; executor weight-version updates run on the caller's
  event loop. Preserve this ownership when updating upstream orchestration.
- Checkpoint finalization is awaited after training and rollout state saves.
  Producer shutdown runs before shared executor resources are disposed.
- Keep additional regression tests in separate files (`test_core_arguments.py`,
  `test_router_readiness.py`, `test_trainer_teardown.py`, and
  `test_trainer_disposal.py`). The corresponding upstream test files are unchanged
  from the recorded comparison base. Small local fixture helpers are intentionally
  independent of upstream test modules.

These hooks do not eliminate compatibility work. When upstream changes the call
order, argument semantics, checkpoint protocol, or rollout lifecycle, review and
retest the interface even if the patch applies without conflict. Generic fixes
belong in shared code and can be proposed upstream separately.

## Preserve deliberate exclusions

`upstream-exclusions.json` records the paths deliberately absent from the
flattened release, plus prefixes for removed components. This is a source-update
policy, not a `.gitignore` file and not permission to skip review of retained
files. Existing notices describe the source actually distributed.

1. In the separate comparison checkout, inspect upstream additions, modifications,
   renames, and deletions since the recorded base. Check both sides of renames.
2. Retain deletions for paths matching the manifest. Skip additions under excluded
   prefixes, including newly named examples and removed model implementations.
   Do not resolve all conflicts with “ours”: retained files need upstream fixes.
3. For `relocated_paths`, review upstream changes for the destination in this
   fork; do not resurrect the old source location or silently discard its fixes.
4. Review additions outside those prefixes and modifications of retained files
   for copied/renamed excluded implementations. In particular, preserve removals
   of the ProRL streaming helper, DeepSeek 3.2 encoder, FlashInfer fake-QAT helpers,
   and MindSpeed/TileLang patches. The rewritten typer utility must not regain
   the removed implementation. A path checker cannot detect these content changes.
5. Run `python tools/check_upstream_exclusions.py --tree /path/to/candidate`.
   It checks tracked and unignored untracked files that currently exist. For
   an initial inventory, use `--paths-from /path/to/changed-paths.txt` instead
   (one repository-relative path per line). Excluded matches exit with status 1.
6. Review any proposed policy change explicitly, update third-party notices where
   needed, and resolve uncertain inclusion questions before importing that code.
   Include the retained exclusions, source target SHA, and validation in the PR.

## Validate an update

Run the exclusion checker and lint, then test the hook contracts, both upstream
and fork regression files, and the affected backend/lifecycle suites. Changes to
publication, checkpointing, training, or dependency versions also need the
relevant GPU smoke/resume runs in the consuming Open Instruct pinned runtime.
CPU contracts alone do not qualify a new upstream runtime. The hooks refactor
itself changes no runtime pin and imports no new upstream version.
