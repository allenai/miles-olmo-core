# Retained-state publication

Core honors `pause_generation_mode=in_place` using the shared engine-pause helper.
The default remains `retract`. In-place pauses active requests, transfers and
finalizes weights, publishes the new version, and resumes without flushing their
attention KV or hybrid recurrent state. It requires disaggregated serving and
`sglang_disable_radix_cache=true`, so new requests cannot reuse old-weight prefixes.
For `--sglang-config`, validation resolves model and server-group settings using
the launch precedence, including eval overrides, and checks every non-placeholder
group whose model receives weight updates. A group cannot re-enable caching over
a safe CLI default. Frozen reference/judge models may still cache prefixes.
The retract path flushes immediately after pausing, before transfer, matching the
shared helper. Engine-drain publication is a separate path and is unchanged.

Retained state is intentionally computed partly under older weights. Original
behavior logprobs and token version spans remain the importance-sampling
contract. `replay_version` is the final serving version, not a promise that all
expert routes were recomputed under that version. Route replay uses the recorded
routes. Existing historical-prefix/latest-forward metric names refer to token
sampling versions; latest-forward tokens can still use retained older state.

Qualification must include hybrid recurrent-state continuation, overlap
scheduling, route alignment, repeated live publications, and fresh requests after
an update. A passing tiny mechanics probe is not evidence of learning stability.
Use a short matched learning run before scaling to three nodes. Inspect the
trainer-versus-behavior mismatch, token-weighted `behavior/*/k3_kl`, clipping and
ESS by response length and age. PPO KL and PPO ESS in refresh mode do not measure
this mismatch. The new KL estimate uses the existing low-var diagnostic bounds;
`k3_kl_saturated_fraction` exposes values affected by those bounds.

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

The GPU runs used this patch before rebase, at `1dbe38b3fe3ffa1eb4b7549b581da00ff1be80e1`.
The application selects in-place mode explicitly in supported learning examples;
the native MILES parser default remains retract.
