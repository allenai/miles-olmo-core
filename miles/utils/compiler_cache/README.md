# Persistent compiler caches

This package owns the existing Core trainer and SGLang compiler-cache facility
moved from Open Instruct. It retains its Core/Olmo settings, source fingerprints,
WEKA default, and environment names. Relocation does not enable it for Megatron
or qualify additional compiler families.

Ordinary integration persists **Triton only**. Every worker uses a private local
cache, restores a verified immutable generation, and publishes after the first
completed training collection, periodically thereafter, and after successful
shutdown. Corrupt or incompatible entries fall back to compilation. Publication
failure does not turn completed training into a failure. CUDA graphs remain
process-local; serving TP greater than one remains excluded from persistence.

The application driver calls:

- `startup_cache.prepare(args, application_root=...)` before creating workers.
  The explicit root contains `open_instruct/` and the runtime lock at
  `build/runtime/miles/runtime.lock.json` (image) or `runtime/miles/runtime.lock.json`
  (checkout). This preserves the original source identity without importing the
  application package.
- `startup_cache.configure_specs(args, specs)` through the worker manager's spec
  transform. Trainer setup hooks and the serving entrypoint install caches before
  loading the training/inference stack. Serving also primes private Hugging Face
  module caches before spawning concurrent importers.
- `startup_cache.publish_progress(args, rollout_id)` after completed collections.
- `await startup_cache.finish(args, success=...)` during cleanup.

Open Instruct retains its `[compiler_cache]` settings and CPU-only input checks.
The default shared root is
`/weka/oe-training-default/open-instruct-compiler-cache/tmp-7d`; absent that mount,
ordinary local caches are used. Custom WEKA roots require an expiry component
such as `tmp-7d`. The default storage cap is 8 GiB per key and the progress
publication interval is 600 seconds. Existing `OI_CORE_STARTUP_CACHE` and
`OI_CORE_CACHE_OBSERVE_ROOT` environment names are intentionally preserved.

`python -m miles.utils.compiler_cache.run --help` exposes the separate experimental
single-node probe, including its broader cache families. Open Instruct's old
`python -m scripts.miles.compiler_cache_run` command forwards to it.

Storage, lifecycle, subprocess, module-priming, and relocation tests live in
`tests/fast/compiler_cache/`. The application keeps configuration and packaged
runtime integration tests. A GPU cold/warm pair is needed to qualify reuse for
a specific image, model, and topology. Moving source files changes fingerprints:
compare cold/warm runs of the same new revision, rather than forcing old entries
to match.

This is a direct relocation. Existing lifecycle state and the large publication
function are retained to avoid changing storage behavior during the move.
