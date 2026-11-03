# Experimental retained-state publication

Core honors `pause_generation_mode=in_place` using the shared engine-pause helper.
The default remains `retract`. In-place pauses active requests, transfers and
finalizes weights, publishes the new version, and resumes without flushing their
attention KV or hybrid recurrent state. It requires disaggregated serving and
`sglang_disable_radix_cache=true`, so new requests cannot reuse old-weight prefixes.
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
