# Retained-state publication

Core honors `pause_generation_mode=in_place` using the shared engine-pause helper.
The default remains `retract`. In-place pauses active requests, transfers and
finalizes weights, publishes the new version, and resumes without flushing their
attention KV or hybrid recurrent state. It requires non-colocated serving and
`sglang_disable_radix_cache=true` by default, so new requests cannot reuse old-weight prefixes.
An opt-in alternative is our plugin backend described in
[Policy-scoped prefix caching](core-policy-cache.md).
For `--sglang-config`, validation resolves model and server-group settings using
the launch precedence, including eval overrides, and checks every non-placeholder
group whose model receives weight updates. A group cannot re-enable ordinary unversioned caching over
a safe CLI default. Frozen reference/judge models may still cache prefixes.
The retract path flushes immediately after pausing, before transfer, matching the
shared helper. Engine-drain publication is a separate path and is unchanged.
Core rejects `abort` mode rather than silently treating it as `retract`; this
affects synchronous configs because fully async runs already rejected `abort`.
Runtime validation runs once at actor initialization. The companion Open Instruct
launcher also rejects colocated layouts and unqualified cache-enabled profiles during config
compilation, before submitting a job.

Retained state is intentionally computed partly under older weights. Original
behavior logprobs and token version spans remain the importance-sampling
contract. `replay_version` is the final serving version, not a promise that all
expert routes were recomputed under that version. Route replay uses the recorded
routes. Existing historical-prefix/latest-forward metric names refer to token
sampling versions; latest-forward tokens can still use retained older state.

The pinned engine has a provenance limitation with overlap scheduling: in-place
pause leaves a launched batch whose result has not yet been processed. The version
bump records output lengths before its token is appended, so that token can be
tagged with the new version despite being sampled under the old weights. For the
qualified non-speculative decoding setup, age and historical-prefix buckets can
therefore misclassify at most one token per in-flight request per publication.
Original behavior logprobs and importance weights are unaffected. Retract drains
the pending result before pausing; this limitation comes from the engine’s
in-place path, not the Core provenance accounting.

Qualification must include hybrid recurrent-state continuation, overlap
scheduling, route alignment, repeated live publications, and fresh requests after
an update. A passing tiny mechanics probe is not evidence of learning stability.
Use a short matched learning run before scaling to three nodes. Inspect the
trainer-versus-behavior mismatch, token-weighted `behavior/*/k3_kl`, clipping and
ESS by response length and age. PPO KL and PPO ESS in refresh mode do not measure
this mismatch. The new KL estimate clamps log ratios to [-10, 10] and the
result to [0, 10]; the existing low-var diagnostic uses [-20, 20] inputs and
[-10, 10] outputs. `k3_kl_saturated_fraction` exposes either bound taking effect.

Publication logs record `pause_mode` and `retained_state`. Preserve failures:
publication exceptions must not resume inference or advance the published clock.

Qualification completed on the B300 hybrid model with 32K responses: a tiny
live-update probe, a short one-node learning run, and a 50-update three-node run
at 3,840 admitted samples. The three-node run had zero stale discards, allocator
retries and OOMs. Mean trainer-versus-rollout KL over updates 25–50 was 0.000178;
long-response historical-prefix token-weighted KL was 0.000143 with no TIS clipping.
These are short-run stability checks, not long-run learning-equivalence evidence.

Runs:

1. Tiny mechanics: https://beaker.org/ex/01M6G1ERM883BMPMGJDHT7446K
2. One-node learning: https://beaker.org/ex/01M6G2BZV92DG7G32Q6BMHGP6T
3. Three-node learning: https://beaker.org/ex/01M6GA8H3EM4B8AG0A725KHZHW

The linked GPU runs used the pre-rebase runtime patch in Beaker image
`01M6G1GR2DG13VFZGAE1TJ3J15`; they did not exercise the final rebased PR head.
The application selects in-place mode explicitly in supported learning examples;
the native MILES parser default remains retract.
