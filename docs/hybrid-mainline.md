# Temporary MainlineLadder bridge

This branch adds Yasha's released Olmo Hybrid (`mainline_ladder`) to the Core
learner and SGLang. It is a temporary correctness bridge, pending GPU qualification.
It does not imply that training or all five scales have passed validation.

The native learner loads the release's paired `olmo-core/v1/.../config.json`,
overrides the full-attention backend, and strictly converts every HF tensor.
Publication streams BF16 tensors under the exact HF names. AdamW parameters stay
FP32. Native policy/optimizer checkpoint save/load reuses the standard Core backend.
Use barrier publication; the MoE-only engine-drain path is rejected.

The serving bridge reuses the pinned release HF/FLA Gated DeltaNet modules,
including `linear_allow_neg_eigval` (`beta = 2 * sigmoid(b)`), FP32 recurrent state,
separate q/k/v convolutions, output gating, embedding scaling, and peri norms.
Full attention uses SGLang RadixAttention with the release QK norms and gate, and
no positional embeddings. All checkpoint names/layouts remain unchanged.

## Supported temporary configuration

- BF16, one inference GPU, TP=1, PP=1, unquantized.
- `MILES_HYBRID_BRIDGE=1` for MILES-managed serving children.
- `SGLANG_EXTERNAL_MODEL_PACKAGE=miles_plugins.hybrid.models`.
- Disable CUDA graphs and radix/prefix caching; chunked prefill must be `-1`.
- No speculative decoding, idle batches, or serving disaggregation.
- GDN state is a per-request-slot Python cache outside SGLang's memory planner.
  Start with `max_running_requests=8` and `mem_fraction_static=0.45` on an A100
  80GB. Slots reset on fresh prefill and states clear when weights are loaded.
  The bridge retains completed-slot state until slot reuse; the high-water slot
  count therefore bounds its persistent allocation. Tune memory only after measurement.
- GDN requests run separately within each packed forward; throughput will be lower
  than a purpose-built packed GDN backend. This is intentional for the initial port.

Standalone server: `python -m miles_plugins.hybrid.serve --model-path CHECKPOINT
--dtype bfloat16 --tp-size 1 --disable-cuda-graph --disable-radix-cache
--chunked-prefill-size -1 --attention-backend triton --max-running-requests 8
--mem-fraction-static 0.45 --context-length 4096` (join as one command).

The runtime must include `transformers_plugin` from scaling-ladders commit
`0fb485ce38733103f7da73973cdf0e74b0fc75db`, plus its FLA dependencies.
The test image keeps the Saturn MILES Torch 2.13 / CUDA 13 / SGLang
`3145136dcd1238754e0ea2b2ffd546532119c71c` toolchain and overlays this fork and plugin.

## Qualification

`python tools/hybrid/qualify_sglang.py --size all --output-dir /results/hybrid`

The matrix tests final midtrain 275m/450m/810m/1.4b/2.7b checkpoints. Each model has
its own subprocess, log, JSON report, and 20-minute timeout; reports survive a later
model failure. Prompts span 1, 3, 31, 127, 511, 2047, and 3071 tokens; mixed batches
perform 32 greedy cached decode steps, repeat requests to exercise slot reuse, and
reload checkpoint weights before repeating. Reference HF teacher-forced logprobs
are compared on the actual served continuation: maximum error <=0.08; at least
95% of generated tokens must be within 0.01 logprob of the HF argmax. These are
explicit BF16 tolerances, not exact-token guarantees. The release FLA fast path is
required; a slow Torch fallback cannot pass qualification.

Reports include checkpoint/config provenance, startup time, batch decode tokens/s,
probability errors, and reload results. A passing matrix qualifies this serving
configuration only. Packed native learner updates, live training publication,
preemption recovery, and learning quality require separate end-to-end GPU checks.

CPU evidence: all 192 actual release tensor names/shapes are covered; unknown
layers and missing tensors fail early. GPU results are pending.
