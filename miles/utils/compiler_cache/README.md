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
a specific image, model, and topology. Moving source files changes fingerprints:
compare cold/warm runs of the same new revision, rather than forcing old entries
to match.

## Core Triton compatibility

Ordinary startup uses the versioned `core-triton-v2` identity. It excludes only
audited non-kernel differences; unknown settings continue to invalidate:

| Input | Core Triton behavior |
| --- | --- |
| Admission budget, HTTP concurrency, global/rollout batch size, samples per prompt, rollout fleet size, policy lag and run deadline | Do not invalidate existing artifacts. Triton's own keys still distinguish any newly encountered kernel specialization. |
| Known single-GPU serving workers | Share a role key across engine slots. Repeated identical visible GPU descriptions collapse to one; mixed hardware remains distinct. Mutable caches stay private to each worker. |
| Trainer ranks, expert/tensor parallelism, microbatch size, packing, precision, kernel backends, compiler controls and unknown settings | Remain part of compatibility. Trainer ranks remain separate. |
| Open Instruct source | Omit tests and the standalone `miles/evaluation` and `miles/distillation` trees. Other source changes, including new modules, still invalidate. |
| Runtime lock | Omit the unused Megatron source entries and patches wholly between those sources. Core/MILES/SGLang pins, unknown sources and patches targeting Core remain significant. |
| Base image, model configuration, GPU architecture/properties, driver and toolchain | Remain significant. |

The separate experimental multi-family wrapper retains its strict identity and
full source fingerprint. Versioning deliberately prevents automatic adoption of
old worker-specific cache keys: the first run of this revision may be cold. A
new runtime image and Open Instruct runtime-pin update are required to deploy it.
CPU tests establish identity and publication behavior; GPU reuse still requires
qualification before treating the draft as a measured startup improvement.

Triton autotuning can record different timing measurements for the same key and
candidate configurations on different workers. Publication retains the first
valid record when only those measurements differ, and continues merging new
kernel artifacts. The receipt counts `reused_autotune_records`. Different tuning
keys, candidates, unknown record formats and conflicting compiled binaries still
reject publication and preserve the last published generation.
