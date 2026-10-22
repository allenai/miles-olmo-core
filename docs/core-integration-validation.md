# Core integration validation

The fork should own the CI check, while Open Instruct owns the consumer test
harness. Keep backend unit and interface tests here; keep synthetic models,
application configs, rewards, preparation, and end-to-end audit code in Open
Instruct. CI should run a pinned consumer harness against the candidate MILES
commit rather than copy that harness into this repository.

## Validation gates

Hosted CPU checks pin public SGLang and Megatron source revisions in
`tests/ci/cpu-dependency-lock.json`. Update these revisions deliberately alongside
compatibility changes; moving dependency branches must not change the result of
an unchanged fork commit. Workflow-dispatch overrides take precedence over PR-body
overrides, then a release lock, then this CPU lock. These are CPU test dependencies,
not the Open Instruct GPU runtime pins. Model-tokenizer checks still download
external assets and can fail independently of source compatibility.

| Gate | Location | What it establishes |
| --- | --- | --- |
| CPU interfaces | `miles-olmo-core/tests/fast/` | Hook dispatch, event-loop ownership, publication acknowledgement and errors, checkpoint ordering, shutdown, and exclusion policy. Required on every change. |
| Local tiny hybrid consumer | Open Instruct `tests/miles/` | One RTX 4090, target-family hybrid Olmo MoE, colocated trainer and serving engine, real generation and nonzero updates, routing replay, checkpoint/export, process restart and resume. |
| Two-GPU trained consumer | Open Instruct `tests/miles/` and `configs/miles/qualification/` | Established trained H045 checkpoint on WEKA, one trainer GPU and one serving GPU, real weight transport, evaluation, checkpoint/resume, export, and clean shutdown. |
| Publication cases | Same consumer harness, separate configs | Barrier, refresh, and engine-drain each exercise their own lifecycle; one successful synchronous run does not cover all modes. |
| Distributed qualification | Dedicated recipes | Multiple training ranks, expert parallelism, and multi-node transport. Two trainers plus a separate serving engine need at least three GPUs. |

Open Instruct's `tests/miles/local_moe.py hybrid` supplies the local architecture
fixture: a dense KDA first block, a KDA latent-MoE block and a full-attention
latent-MoE block, including shared experts and the target gating/normalization
settings. Reduced dimensions fit a colocated 4090 run. The conventional tiny
trained MoE checkpoint and the Qwen `smoke.py` fixture do not cover those hybrid
components; retain them only as additional cases. Explicit compatible serving
rounding is needed for the reduced geometry, which does not match the production
automatic profile. Disable zero-variance reward filtering for the random model;
router auxiliary losses provide an update signal.

Keep toy models local. Use the established trained H045 checkpoint on WEKA with
`configs/miles/qualification/h045.toml`, introduced in Open Instruct PR #1987,
with bounded generation, fixed data and seeds, and explicit checkpoint/evaluation
cadence. The fork-specific GPU CI workflow remains proposed follow-up work, not
an already enabled CI gate.

The existing toy script calls Open Instruct's Python API. A maintained structured
CLI recipe should additionally cover configuration compilation and preparation;
passing the API fixture alone does not establish those entrypoints work.

## Reproducibility contract

Freeze the binary image, MILES candidate, OLMo-core and serving sources, consumer
harness revision, fixture contents, model/tokenizer configuration, seeds, dtype,
attention backend, and batching. A self-contained synthetic fixture avoids
mutable downloads and gives nonzero gradients without depending on a random
model answering GSM8K correctly. Retain the real-data recipe as a separate smoke
check.

Use exact assertions for provenance, sample identities, prompt cursor, completed
steps, checkpoint completeness, and policy-version ordering. Check finite,
nonzero gradients and parameter changes, serving/training logprob agreement,
weight publication acknowledgement, and HF export reload. Compare repeated runs
on the same GPU type before setting numerical tolerances. Fixed seeds alone do
not establish bitwise determinism across hardware or asynchronous schedules.
Greedy generation can simplify output comparisons, but identical responses in a
GRPO group can remove the learning signal; the synthetic reward must still
exercise optimization.

A resume case must verify optimizer/scheduler and data state, not merely that a
checkpoint loads. Compare a resumed run with uninterrupted execution where the
mode supports that contract. Refresh coverage also needs evidence that retained
samples actually cross a policy boundary; selecting the mode is insufficient.
Engine-drain coverage needs ownership/version transitions and released snapshots.
Failure cases must check that incomplete saves are not advertised as committed
and failed publication does not admit an unacknowledged engine.

## CI responsibility

A workflow in `miles-olmo-core/.github/workflows/` should select the candidate SHA,
check out an immutable Open Instruct harness revision, set that candidate in the
consumer runtime lock, build the image, and invoke the consumer's gate. Verify
both the source SHA and the imported module path inside the container so the
check cannot accidentally exercise a previously installed MILES package.

The consumer harness owns assertions and machine-readable results. The fork's
workflow owns triggers, execution resources, status, and artifact publication.
Record the resolved lock, consumer SHA, image digest, fixture/config hashes, GPU
type, logs, audit JSON, and separate startup, steady-state, checkpoint/evaluation,
and shutdown timings. A GPU gate must fail explicitly when its required runtime
or device is missing rather than report success through skipped tests.

Start with manually triggered qualification and a bounded timeout. Promote the
measured local/two-GPU cases to required checks once repeatability and cost are
known. Keep distributed and publication-mode coverage visible as separate checks;
do not label the basic two-GPU recipe as exercising every integration interface.

## Optional consumer group

The isolated Open Instruct companion provides
`tests/miles/qualification/runner.py --case lifecycle|resume|all`. It is an explicit
command, outside ordinary pytest discovery, and does not launch cluster jobs.
The local fixture JSON fixes a three-layer hybrid architecture, seed, tokenizer
construction and three synthetic arithmetic prompts. Preparation is offline.
The lifecycle case runs three updates with four responses per prompt; the resume
case compares it with two updates followed by a fresh-process third update.

The audit covers routing replay, policy versions, finite nonzero gradients and
updates, complete checkpoints, data cursor, serving/training score agreement and
exact export reload. The comparison includes all saved model and Adam state,
scheduler state and data state. Exact tokens/routes and `rtol=1e-5, atol=1e-6`
for floating optimizer state are acceptance thresholds to validate on the
intended hardware. Do not relax a failed threshold without investigating it.
Greedy decoding and router auxiliary losses make this a mechanics/resume test;
it does not establish task learning or sampling-RNG continuity.

Missing runtime or GPU is an explicit failure. Each subprocess has a bounded
timeout. Results include imported paths, source-content hashes, fixture hashes,
the launcher-supplied immutable image ID, command logs and comparison metrics.
Keep this group opt-in until repeatability and runtime cost are established.
The trained H045 WEKA gate remains separate; a local pass does not qualify it.

Adam first/second moments use the stricter absolute tolerance `1e-12` (relative
tolerance `1e-5`), since small moments must not compare equal to a reset optimizer.

### Initial local evidence for the v0.1.2 consumer companion

The optional self-contained group completed on one RTX 4090 in 416 seconds.
Three uninterrupted updates matched two updates plus a fresh-process third update:
all generated tokens/routes, scheduler/data state and 560 floating checkpoint
tensors matched exactly (maximum absolute difference zero). Both arms processed
12 responses and passed generation, nonzero gradient/update, routing, publication,
checkpoint and exact export/reload audits. Fixture preparation repeated with
identical file hashes. These are mechanics checks using auxiliary losses, not
learning evidence.

A separate fresh-container control differed numerically after its first update
and later produced some different greedy tokens. Its cause is not established;
the successful paired resume comparison does not establish cold-start or
cross-build reproducibility. Keep the group opt-in, preserve strict comparisons,
and characterize that variability before defining a golden numerical baseline
or making this a required GPU CI gate. The trained H045 two-GPU gate remains
unrun for this source update. A source-mounted local test image does not replace
building and qualifying the eventual committed consumer runtime lock.
