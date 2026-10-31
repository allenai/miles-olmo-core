# Core runtime maintenance

Keep the backend maintenance policy in this repository. An Open-Instruct runtime
upgrade should change its existing source lock and Dockerfile, integration calls,
parser snapshot and regression tests. Avoid adding another dependency manifest,
package validator, release-receipt format or qualification framework there for
each upgrade. Put run receipts and troubleshooting logs in PR evidence or
artifacts. [Integration validation](core-integration-validation.md) describes
backend-owned qualification and the separate consumer boundary.

## Updating the runtime

1. Follow [the filtered upstream update procedure](upstream-updates.md) and
   preserve the exclusion policy. Do not attach raw upstream ancestry.
2. Review the target release's source interfaces and Docker/dependency recipe
   together. A new SGLang checkout can require different Python and compiled
   packages even when the old image still imports successfully. Audit the source
   requirements and the selected packages' transitive requirements, including
   CUDA extras. `--no-deps` does not install an extra's dependencies; verify what
   the pinned base already provides. Import the serving engine as well as the
   top-level package, but do not treat successful imports as dependency validation.
3. Update the consumer's existing source lock and image definition. Keep the
   established base/build stages unless a required dependency change justifies
   replacing them. Do not use an old application image as a new base merely to
   avoid rebuilding a cached stage.
4. Adapt consumer calls to the new interfaces and regenerate its parser snapshot
   using those sources and installed serving packages. Review removed/renamed
   options; a successful Git merge is not an interface compatibility check.
5. Run the affected CPU and runtime tests, build the committed image, and select
   GPU checks appropriate to the changed boundary. Record which exact image and
   revision each result tests. A previous image's success is supporting evidence,
   not proof that a later image has been exercised.

Keep dependency fixes in the image recipe. Use behavioral regression tests to
cover observed failures rather than maintaining a second dependency resolver or
checks tied to private kernel symbol names. Preserve existing serving choices
when updating packages; changing the package need not force every run onto one
GEMM backend.

## v0.1.2 integration changes

The merged fork is `9ea6759fd872e78516decc45a684183f77918e14`, tagged `v0.1.2`.
The Open-Instruct consumer pins SGLang `b84bec5761a6cdc8b1df86b9b89ead80a7a9bb02`,
olmo-sglang `109f881dc1fa8acdb16bc2717a910f01958d79c9`, and unchanged OLMo-core
`7e97bc43ff25d080ae4fd0eab1050a79caf80e55`.

The consumer must adapt the training-model/weight-publication call signatures,
use the new manager interface instead of calling `.remote()` itself, and import
the moved HF config module. Initialize local Ray before W&B creates its local
service connection, then initialize tracking before serializing worker args.
Otherwise offline workers can inherit the driver's W&B service or miss its run
ID. Preserve an already-connected Ray cluster. The rank-logging fix in
[olmo-sglang PR #8](https://github.com/allenai/olmo-sglang/pull/8) uses the
initialized distributed process group's rank; it does not change model math.

### Serving packages

| Package | Required update for the exercised runtime |
| --- | --- |
| `sglang-kernel` | `0.4.7` |
| `flashinfer-python[cu13]` | `0.6.18`, matching the pinned SGLang source requirement |
| `flashinfer-cubin` | `0.6.18`, matching FlashInfer Python |
| `sglang-router` | Exact upstream wheel below; version `0.3.2` alone is insufficient |

The previous FlashInfer package lacked a Blackwell BF16 kernel imported by the
new SGLang path. Updating both FlashInfer packages allowed automatic BF16 backend
selection to run on Blackwell. The existing Torch selection remains available.
A subsequent source-requirement audit found nine additional missing or mismatched
serving packages in the inherited base. Open-Instruct PR #2047 aligns these in
the existing Dockerfile, including Cython required by nvshmem4py. The audit of
SGLang requirements and the changed packages' direct requirements passed for
image `01M5JW54NEVGNZES1SY2FKY63A` (application `b32020aee`). This does not
qualify every optional backend or substitute for a GPU run.

The router wheel is the `sglang_router-0.3.2-cp38-abi3-manylinux_2_39_x86_64.whl`
asset from the upstream [cu130-torch213-x86_64 wheel bundle](https://github.com/radixark/miles-wheels/releases/tag/cu130-torch213-x86_64),
with SHA-256
`28fede513319fcff4a38e5d859589498799c2e2861cc25d78ed371b3aaf3d192`.
Its source is `radixark/sgl-router-for-miles` commit
`a0cf7da533105d8d8da13119700934fb0c6e10d7`, containing fix
`8ecf98086b15942b097a212006dc138ed4252751`. The release URL and version are reused;
keep the content hash. These are the inspected current bundle's bytes, not
necessarily the original v0.1.2 release bundle's bytes.

That fix skips automatic tokenizer registration for HTTP workers unless the
router explicitly receives a model/tokenizer path. Token-ID generation does not
need a router-local tokenizer. The old wheel attempted to parse Transformers'
`{% generation %}` tags with MiniJinja and logged an error. Preserve those tags
in checkpoint templates: they describe training spans. The fix avoids the
unnecessary parsing; it does not add support for those tags to MiniJinja.
The manual router test fails with the old wheel and passes with the replacement.

## Core refresh response decoding

Core refresh requests with routing replay ask the distributed HTTP actor to
decode the base64 routing blob before returning the response through Ray. It
returns a flat `int32` array; the producer reshapes it using the accumulated
token count and model layers, retaining nonzero and policy-provenance checks.
Ordinary POST callers keep their existing response contract unless they opt in,
and raw-byte responses bypass decoding. The local HTTP fallback remains supported.

Worker observations include `replay_decode_seconds`,
`replay_decode_thread_cpu_seconds`, and `replay_decoded_bytes`. Decoding occurs
after the HTTP retry loop, so malformed replay cannot cause a delivered request
to be resent. It is synchronous within each HTTP actor; this change does not add
a background decoding executor, bounded payload fetching, or compact token/logprob
transport. Ray reconstruction and producer batch preparation still need separate
measurement. CPU transport tests do not establish GPU throughput gains.

## Recorded validation, October 22, 2026

These results are historical evidence at specific revisions. They do not make
the removed consumer qualification scripts a maintained API or a required CI gate.

| Check | Image/revision and result |
| --- | --- |
| Trained H045, two Holmes B300 GPUs | [Beaker run](https://beaker.org/ex/01M5JE15RBBZMZTKJ5ASMM1XGS), image `01M5JDNZ4CC08QQQAB6SHCP4Y8`, application `0646b575b90bf7347d1096c68950a3b1721e6908`: two optimizer steps, weight publication/comparison and offline W&B shutdown; exit 0 in about 14 minutes. Predates the router replacement. |
| Saturn CPU/WEKA tracking | [Beaker run](https://beaker.org/ex/01M5JFEF9C14SMTF7WYY4DQBVM), same image; immutable test payload supplied after the build: 94 tests passed in 43.20 seconds. CUDA hidden; no A100 training or online W&B claim. |
| Local RTX 4090 after router replacement | Application `6b538afd4`, Docker `sha256:3df073e834ec9ab1bed30a1c6333768ee6d3b8c4c422851bbd13c74f6b566d7e`: 425 seconds, three updates versus two plus resumed third, all 560 model/Adam tensors equal, both exports exact, maximum train/serving log-probability error 0.01161 against 0.05. |
| Embedded CPU-compatible runtime suite | Image `01M5JR2WB3AEW2M9PHADKFJKGA`, application `7aeaf73cde16d089f67bcb146c8da898f7310c9b`: 596 passed, 57 skipped; GPU-only OPD attention excluded. Only tests changed from the preceding GPU-tested image. |

The H045 run produced 16 responses from eight prompts; all reached the 128-token
limit and received zero task reward. Entropy gradients exercised optimizer steps.
The local toy used auxiliary losses. Neither establishes learning. The final
router image did not repeat H045 on Blackwell, and these checks do not qualify
multi-node operation or every publication mode.

Open-Instruct PR #2047 subsequently removed the qualification additions and
simplified its image recipe. New builds must identify their own source/image
and validation; do not carry the historical image IDs forward as if they tested
that revised recipe.

The reviewed rebuild above passed 606 CPU-compatible runtime tests (57 skipped)
and the standalone real-router regression. The existing packed-attention GPU
test, with its moved Miles imports corrected, failed on an RTX 4090: Transformer
Engine selected an FA4 kernel requesting 131072 bytes of shared memory, above
the device limit of 101376 bytes. That FA4 path remains unresolved; the later
Blackwell check below uses the established Torch attention settings.

The same image subsequently passed the trained H045 two-GPU Blackwell check:
[Holmes run](https://beaker.org/ex/01M5JWY055PD11Z6R06W64CJF2), unallocated high
priority in `ai2/olmo-instruct`, with a 30-minute execution timeout. It exited 0
after 13m59s, completed two optimizer steps and 16 responses from eight prompts,
published weight versions 0/1/2, and passed three exact serving-weight comparisons.
All 16 driver timing stages passed. Driver, trainer and rollout W&B logs recorded
exit code 0. Mean train/serving log-probability differences were 0.02459 and
0.03273 against the configured 0.05 mean limit; tokenwise maxima were 0.86104 and
0.83716. This is not a per-token 0.05 guarantee. All responses were truncated at
128 tokens with zero task reward; entropy supplied finite, nonzero gradients.

Serving startup took 303s, trainer startup 60s, the first training stage 284s
including first-use overhead, and the second training stage 6.03s. Publication
stages with equality diagnostics took about 22s each. The image pull took 7m55s
before application startup. Nonfatal SGLang `freeze_gc` connection-refused and
warm-up health-check messages occurred before serving recovered. No checkpoint,
resume, multi-node, OPD/FA4 or online W&B qualification is implied.

The same H045 recipe on an A100 80GB
([Saturn run](https://beaker.org/ex/01M5JWVBD3J2306QHW5229C9MA)) passed serving
startup but exhausted trainer GPU memory while allocating optimizer state,
before any training step. Its driver W&B log correctly recorded exit code 1,
and the job exited 1. An A100 gate needs a smaller trained checkpoint; reducing
response length alone does not fix this optimizer-state capacity limit.
