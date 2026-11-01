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
  (checkout). This fingerprints the Core application source without importing the
  application package; exclusions are described below.
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
a specific image, model, and topology. Legacy strict scopes hash source locations; the ordinary Triton policy below
separates source provenance from artifact compatibility.

## Core Triton compatibility

Ordinary startup uses `core-triton-v3`. Compiler artifacts and experiment
provenance have different identities. The full application/runtime source hashes
and lock remain in each worker receipt as `provenance`; Python-only revisions do
not invalidate every compiled kernel. Triton checks its own source/dependency,
signature, specialization, backend and compiler-option keys when choosing each
artifact. Changing a kernel still compiles a new specialization.

The outer compatibility key retains the binary base, actual device/toolchain,
the installed Triton source/binaries, native source and extensions in the
application, MILES, OLMo-core, olmo-sglang and SGLang, model configuration,
parallel geometry, compiler environment and unknown settings. Known non-kernel
settings such as lag, batch size, prepared-batch overlap, layer timing and
truncated-response reward gating are excluded. Trainer ranks remain separate;
known equivalent TP1 serving slots continue sharing their role key. Private
mutable directories, archive/file verification and collision rejection remain.

This scope applies **only to Triton**. It must not be used to restore Inductor,
FA4/CuTe or other compiler families. Legacy v2 archives lack the compiler/native
identity evidence, so v3 does not automatically import them. The first v3 run
may be cold; subsequent Python orchestration changes should reuse its artifacts.

With diagnostics enabled, `compatibility_explanation` compares the current
identity with the last observed startup for the same model and worker slot.
It reports changed field names, previous fingerprint and whether provenance
changed. This bounded diagnostic pointer never selects an archive and is not
proof that the previous run published successfully. Concurrent starts can change
which observation is compared. Corrupt diagnostics cannot prevent ordinary
restoration. Native cache decisions still come from the exact key and checksums.

The optional Triton compilation listener adds `compile_hit` / `compile_miss`
events with kernel names, compiler microseconds and lowering-stage timings to
`activity-PID.jsonl`. These differ from `group_hit` / `group_miss`, which count
file-cache operations. Existing compilation listeners are preserved. CUDA graph
capture, module loading and non-Triton compilers are outside these timings.

`python tools/probe_triton_cache.py --output /output/cache-probe` is a bounded,
single-GPU fresh-process qualification: cold, warm, changed Python provenance,
changed kernel dependency, changed shape, and FA4 cold/warm. It exercises KDA
forward/backward and SGLang expert inference, checks numerical outputs, and
requires no Triton compilation in the unchanged warm phases. TileLang artifacts are also archived and restored into fresh directories, with
its supported JIT diagnostics checking for unexpected warm compilation. These
probes do not enable additional compiler families by default.

Triton autotuning can record different timing measurements for the same key and
candidate configurations on different workers. Publication retains the first
valid record when only those measurements differ, and continues merging new
kernel artifacts. Different tuning keys/candidates or conflicting binaries reject the entire
incoming kernel group. The previously published group stays byte-for-byte
unchanged; unrelated new kernel groups can still be published. Receipts list
`rejected_incoming_groups` and startup logs warn about them. Cross-group metadata
remains a strict rejection; other compiler families retain strict publication.

## Opt-in TileLang lifecycle

Open Instruct's `core.compiler_cache_tilelang` enables a separate
`core-tilelang-v1` compatibility envelope. It fingerprints installed TileLang
Python/native/compiler sources, including its bundled TVM, in addition to the
base runtime, device/toolchain and compile settings. Native-library stamping is
forced on. This is opt-in pending GPU output/gradient qualification.

The mutable cache and compiler scratch are separate private directories. Active
`.staging` exports and changing snapshots are rejected; publication retries on
a later interval. Independent TileLang receipts and storage caps prevent its
state from being mistaken for Triton hits. Diagnostics use TileLang's supported
`TILELANG_JIT_DIAGNOSTICS` phase logging.
