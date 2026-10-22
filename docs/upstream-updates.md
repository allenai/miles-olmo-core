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
  Ray methods. They explicitly opt out of the typed HTTP RPC schema; Core
  currently requires Ray for both placement and worker communication. They
  delegate to Core helpers. Producer mutations still execute
  on the producer event loop; executor weight-version updates run on the caller's
  event loop. Preserve this ownership when updating upstream orchestration.
- Checkpoint finalization is awaited after training and rollout state saves.
  Producer shutdown runs before shared executor resources are disposed.
- Keep additional regression tests in separate files (`test_core_arguments.py`,
  `test_router_readiness.py`, `test_trainer_teardown.py`, and
  `test_trainer_disposal.py`). The corresponding upstream test files are unchanged
  where possible. Fork regression files may reuse upstream test setup, but keep
  their additional assertions separate.

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

### Audit history and content before publishing

A passing path check is necessary, but does not establish that removed source
has stayed removed. Complete this review before pushing the candidate, including
before opening a draft PR:

- Inspect intermediate upstream commits, not only the comparison-base and target
  trees. Code can be added under an excluded directory and moved elsewhere before
  the release; neither endpoint then exposes its former location. Use
  `git log --follow --name-status` for suspect destinations and inspect the
  kernel/model consolidation commits with rename and copy detection enabled.
- Compare candidate content with excluded implementations at the old base, target,
  and intermediate revisions. Similarity searches are review aids, not proof of
  independent origin. Review extracted functions and partial-file adaptations as
  well as whole-file moves. Distinguish common launcher boilerplate and explicitly
  permitted relocations from excluded implementation code.
- Check retained Dockerfiles, scripts, imports, tests, and documentation for
  references to removed files. Preserve omissions in every newly added Docker
  variant. Do not restore an excluded patch just to make a build succeed.
- Apply the existing source-patch exclusions to newly added adaptations from the
  same excluded component. External dependencies and distributed source patches
  have different scopes; permission to install a dependency does not establish
  permission to restore its omitted source adaptation.
- Update third-party notices for retained code that moves and newly retained
  adaptations. Record the evidence, removed destinations, permitted relocations,
  and unresolved questions alongside the update. Do not equate this engineering
  audit with a comprehensive licensing determination.

The v0.1.2 rehearsal initially missed three DeepSeek V4.1 kernels moved to generic
paths: `hyper_connection/mhc.py`, `position/rope.py`, and `quant/fake_quant.py`
under `miles/kernels/`. Intermediate history exposed their origin. The follow-up
also removed a newly added TileLang source patch and stale TileKernels patch
references in the CUDA 12 and Rubin Dockerfiles. These findings belong in the
exclusion manifest and future content reviews, not only in a one-off checklist.

## Validate an update

Run the exclusion checker and lint, then test the hook contracts, both upstream
and fork regression files, and the affected backend/lifecycle suites. Changes to
publication, checkpointing, training, or dependency versions also need the
relevant GPU smoke/resume runs in the consuming Open Instruct pinned runtime.
CPU contracts alone do not qualify a new upstream runtime. The hooks refactor
itself changes no runtime pin and imports no new upstream version.

Run the complete CPU selection as well as focused integration regressions. Use
bounded subprocesses or workers so a hung test produces a reported failure rather
than preventing the rest of the panel from running. Record the exact selection,
dependencies, timeouts, failures, and skips. A timed-out sweep is incomplete;
standalone reruns do not erase failures seen in the combined suite.
Mirror the dependency setup and source-path environment in
`.github/workflows/_run-cpu-ci.yml`; inspect skip reasons explicitly. In this
rehearsal, missing Helm skipped hundreds of CPU chart tests, and missing mbridge
skipped adapter tests. Record any remaining optional-backend exclusions rather
than calling such a run “all CPU tests passed.”

Then exercise the candidate through the consuming Open Instruct driver on a
local 4090 with the tiny target-family hybrid-MoE fixture, followed by the trained
H045 two-GPU smoke/resume gate. Record actual imported source paths and revisions
so an older installed package cannot accidentally pass the candidate's checks.
Do not replace the WEKA recipe with the random tiny fixture, or describe a GPU
import check as a training lifecycle pass.

See [Core integration validation](core-integration-validation.md) for the proposed
local, two-GPU, and distributed consumer gates and their repository ownership.

## v0.1.2 source update

Target: `a22fd824191c64ba7c5d6c531f125406e4ce9cc6`. This is a reviewed source-only
import prepared in an isolated local comparison checkout. The comparison base
advances for subsequent source updates; no upstream ancestry is attached. Runtime
qualification is separate and remains pending the consumer gates below.

- Weight publication now opens/closes its inference window in
  `placement_group.update_weights`. The trainer accepts `UpdatableEngines`; it
  must not independently close that same window. Failed publication releases
  the window without admitting engines.
- Core's `publication/transport.py` adapts already-exported HF buckets to the
  upstream broadcast/IPC protocols. Core retains session/version ownership,
  and IPC acknowledgement completes before a bucket can be reused.
- Worker creation uses providers and typed handles. Open Instruct needs a
  companion driver update for the new construction, orchestration, and disposal
  APIs before its pinned runtime can advance. Core remains single-policy and
  single-trainer-cell; Kubernetes/HTTP RPC are not qualified for this backend.
- Preserve the current-forward versus rollout diagnostic semantics, even when
  PPO uses a stored reference. Frozen loss/gradient snapshots cover this choice.
- Keep session streaming disabled. Preserve upstream response-intent handling
  and sampling-mask support for nonstreaming requests.
- Retain deliberate exclusions when implementations move under `miles/kernels`,
  and exclude newly added descendants of removed model families and tests that
  require absent example/model code. The manifest records the exact paths.
- The rewritten dataclass CLI gains environment loading and tuple conversion
  without restoring the deliberately removed implementation.

Qualification still requires a consumer runtime build, the local tiny hybrid-MoE
gate, and the established trained H045 two-GPU smoke/resume gate. Previous PR #3
GPU runs do not validate this release update.
