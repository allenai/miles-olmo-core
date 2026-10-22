# Core integration validation

Backend qualification belongs in miles-olmo-core. Keep its model fixtures,
recipes, assertions, maintenance notes, and eventual CI entrypoints here and
self-contained. Open-Instruct owns its application integration: configuration
compilation, preparation, rewards, driver ordering, and the image definition
that consumes the pinned backend. Its upgrade PRs should update those integration
points and their regression tests.

The earlier proposal to maintain the backend qualification harness and H045
recipe in Open-Instruct is superseded. Those additions were removed from
[Open-Instruct PR #2047](https://github.com/allenai/open-instruct/pull/2047).
There is not yet a self-contained hybrid-MoE GPU qualification harness in this
repository. Do not treat the historical consumer scripts as an enabled CI gate.
See [runtime maintenance](core-runtime-maintenance.md) for the v0.1.2 dependency
changes and validation evidence.

## Validation levels

| Level | Ownership and scope | Status |
| --- | --- | --- |
| CPU backend interfaces | This repository's `tests/fast/`: hook dispatch, event-loop ownership, publication errors, checkpoint ordering, shutdown, exclusion policy | Existing tests |
| HTTP router compatibility | This repository's `tests/manual/runtime/test_core_http_router.py`: real router, local fake HTTP engine, registration, token-ID forwarding, removal | Explicit CPU runtime test; no GPU or Open-Instruct dependency |
| Local tiny hybrid MoE | Backend-owned model fixture and runner: generation, nonzero updates, routing replay, checkpoint/export and paired resume on one 4090 | Historical consumer evidence; self-contained backend harness still to implement |
| Trained two-GPU integration | Backend-owned trained H045 recipe: one trainer and one serving GPU on WEKA | Bounded startup/training evidence; maintained backend recipe still to implement |
| Publication and distributed modes | Separate cases for refresh, engine drain, expert parallelism and multi-node transport | Must be qualified separately |

Hosted CPU checks pin public SGLang and Megatron sources in
`tests/ci/cpu-dependency-lock.json`. These are CPU-test dependencies, not the
consumer's GPU image pins. Keep those roles distinct when reviewing an update.

Run the router check inside a runtime containing pytest, requests, tokenizers
and the pinned sglang-router wheel, from the fork root:

```bash
python -m pytest --confcutdir=tests/manual/runtime \
  tests/manual/runtime/test_core_http_router.py -q
```

The configuration boundary excludes unrelated parent test fixtures. This test
starts only localhost processes, uses an offline tokenizer fixture, and fails
if the required router runtime is missing. It is intentionally outside default
CI selection until the matching binary dependencies are available there.

## Requirements for a backend GPU harness

Build the tiny target-family model locally: a dense KDA block, a KDA latent-MoE
block, and a full-attention latent-MoE block, with shared experts and the target
gating/normalization. Use a local tokenizer, fixed inputs and seeds, reduced
geometry, and explicit compatible serving rounding. Keep toy models local;
use the established trained H045 checkpoint for WEKA tests. Pass checkpoint,
output and placement details as inputs rather than embedding personal scratch
paths or a particular workspace in a shared recipe.

The backend harness must not import Open-Instruct or require its checkout.
Consumer end-to-end checks remain a separate boundary. Starting the backend
through its Python API does not qualify Open-Instruct's configuration CLI,
dataset preparation, rewards, or orchestration.

Check provenance, sample identities, prompt cursor, policy-version ordering,
finite nonzero gradients and updates, weight publication acknowledgement,
checkpoint completeness, score agreement and exact export reload. Test failed
publication and incomplete saves separately. Compare a resumed execution with
an uninterrupted control, including model, Adam moments, scheduler and data
state. Historical comparisons used `rtol=1e-5, atol=1e-6`, with `atol=1e-12` for
Adam moments to distinguish small saved values from a reset optimizer. Investigate
failures before changing tolerances.

Greedy responses and auxiliary losses can exercise update mechanics without
successful task answers. They are not learning evidence or sampling-RNG
continuity tests. Fixed seeds and one successful paired resume do not establish
cross-container or cross-hardware determinism; a previous fresh-container
control differed after its first update. Establish repeatability before choosing
a golden numerical baseline or making a GPU gate required.

Start with an explicit opt-in command and bounded process timeouts. Missing
runtime/GPU prerequisites must fail rather than silently skip a requested gate.
Record exact source revisions, imported paths, image identity, fixture hashes,
GPU type, logs and comparisons in run artifacts. Report startup, sustained
training, checkpoint/evaluation and shutdown timing separately. CI should invoke
this repository's own harness once implemented; new CI automation is not part
of the v0.1.2 consumer upgrade.
