# Core integration validation

The fork should own the CI check, while Open Instruct owns the consumer test
harness. Keep backend unit and interface tests here; keep synthetic models,
application configs, rewards, preparation, and end-to-end audit code in Open
Instruct. CI should run a pinned consumer harness against the candidate MILES
commit rather than copy that harness into this repository.

## Proposed gates

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
the two-GPU structure in `configs/miles/examples/small.toml`, bounded generation,
fixed data and seeds, and explicit checkpoint/evaluation cadence. The
qualification config directory and fork-specific GPU CI workflow above are
proposed follow-up work, not an already enabled CI gate.

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
