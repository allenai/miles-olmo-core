# Policy-scoped prefix caching

Repeated long prompts make retained-state RL expensive when every response repeats prefill. This opt-in integration combines our Olmo-SGLang policy-cache backend with existing SGLang routing and scheduling features. All changes are in AllenAI repositories; SGLang remains pinned at `b84bec5761a6cdc8b1df86b9b89ead80a7a9bb02`.

Use `--pause-generation-mode in_place`, enable radix caching, select `--sglang-radix-cache-backend olmo_policy_versioned_v1`, `--sglang-router-policy consistent_hashing` and `--sglang-schedule-policy lpm`. Install the companion `allenai/olmo-sglang` branch `codex/policy-prefix-cache`. An older plugin cannot resolve this backend. Retained-state publication remains non-colocated, and the plugin initially supports only device-only TP1 text generation.

The refresh transport previously omitted the ordinary routing headers. It now preserves explicit routing keys and, for consistent hashing, supplies a stable `prompt:{group_index}` key when missing. Siblings share this key while retaining distinct request IDs. Other policies still use the shared routing helper. This routing fix is independent of enabling the cache.

Before publication, Core checks each updated engine's effective cache backend and current weight version through the existing server-info API. It pauses, begins and finishes the weight update, publishes the completed-step label, verifies the committed label, and then resumes. Failed readback leaves generation paused. Cache-off defaults take no extra readback RPCs.

The plugin scopes new requests to that committed policy label and keeps retained old/mixed-policy state private until released. Labels must increase whenever weights change; republishing the same label is valid only for unchanged weights. No custom engine endpoint or transaction epoch was added. The cache uses normal bounded eviction, and existing behavior-logprob/version-span accounting is unchanged.

The initial Open Instruct integration uses its generic profile with explicit concurrency and recurrent-cache capacity. Built-in LPM in-batch prefix scheduling can prioritize one cold sibling, but does not guarantee one prefill under every scheduling condition. Measure cache reuse and throughput in a GPU probe before using it for LC training.

Focused CPU tests cover publication ordering and failure, resolved cache settings, routing headers and the real refresh HTTP transport. The companion plugin tests real CPU KV/KDA allocation and cleanup against the pinned engine.

A two-B300 DocQA integration probe completed four optimizer updates with exit code 0, cache hits, and five retained-state publications (versions 0–4). Thirty-two logged responses crossed policy boundaries while serving. All four consumed training batches had already been generated under version 0 during first-update compilation, so the probe did not exercise training consumption of cross-version responses. Fresh-policy cold-engine parity, repeated-update memory stability and a cache-off throughput comparison remain pending. No speedup is claimed.

Runs:

1. Four-update integration probe: [Beaker](https://beaker.org/ex/01M6Y5Q3BWZQVRVN1CG0M7NNHM), result dataset `01M6Y5Q3C5VM98YMG402TE8WSY`. Tested MILES `0c2b405`, plugin `f0444a8`, application `a2feaaef3`, Core `903e57d`, image `01M6Y5NN3FMQGCVV2WX92AMJS6`. The tested integration branch also contained separate eager-scoring and startup-failure fixes; this PR extracts only cache publication and sibling routing. The separated PR head has CPU regression coverage, not a new GPU qualification.
