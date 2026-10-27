# Temporary MainlineLadder bridge

This branch adds Yasha's released Olmo Hybrid (`mainline_ladder`) to the Core
learner and SGLang. It is a temporary correctness bridge with qualification reports described below.
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
  The Rhea A6000 tests use `max_running_requests=4` and
  `mem_fraction_static=0.35`. Slots reset on fresh prefill and states clear when weights are loaded.
  The bridge retains completed-slot state until slot reuse; the high-water slot
  count therefore bounds its persistent allocation. Tune memory only after measurement.
- GDN requests run separately within each packed forward; throughput will be lower
  than a purpose-built packed GDN backend. This is intentional for the initial port.

Standalone server (qualified fallback): `MILES_HYBRID_REFERENCE_BACKBONE=1 python -m miles_plugins.hybrid.serve --model-path CHECKPOINT
--dtype bfloat16 --tp-size 1 --disable-cuda-graph --disable-radix-cache
--chunked-prefill-size -1 --attention-backend triton --max-running-requests 4
--mem-fraction-static 0.35 --context-length 4096` (join as one command).

The runtime must include `transformers_plugin` from scaling-ladders commit
`0fb485ce38733103f7da73973cdf0e74b0fc75db`, plus its FLA dependencies.
The test image keeps the Saturn MILES Torch 2.13 / CUDA 13 / SGLang
`3145136dcd1238754e0ea2b2ffd546532119c71c` toolchain and overlays this fork and plugin.

## Reference-backbone fallback

`MILES_HYBRID_REFERENCE_BACKBONE=1` selects a slower fallback: each scheduled
request runs the complete pinned release HF backbone with SDPA and its own hybrid
cache. SGLang still owns scheduling, sampling, logprob reporting, and weight-load
APIs. This removes packed backbone GEMM and attention-backend differences from
the comparison. The RadixAttention path remains available with the variable unset.

The fallback keeps both GDN state and full-attention KV outside SGLang's allocator;
its preallocated KV pool is unused. Keep the conservative four-request/0.35 memory
settings. It uses per-request forwards and growing HF KV tensors, so throughput
and allocation behavior differ substantially from a native packed SGLang model.
All existing eager/TP1/no-prefix/no-chunking restrictions still apply.

Qualification command: add `--reference-backbone` to the BF16 matrix.
If separately testing `--fp32-logits`, also set `core.fp32_lm_head=true` in MILES
so learner and serving output precision agree. The output-head flag alone did not qualify the RadixAttention path at every
scale. This fallback is intended to unblock temporary experiments, pending the
same numerical and reload checks, not replace an optimized inference backend.

## Qualification

`python tools/hybrid/qualify_sglang.py --size all --output-dir /results/hybrid`

The matrix tests final midtrain 275m/450m/810m/1.4b/2.7b checkpoints. Each model has
its own subprocess, log, JSON report, and 20-minute timeout; reports survive a later
model failure. Prompts span 1, 3, 31, 127, 511, 2047, and 3071 tokens; mixed batches
perform greedy cached decode, repeat requests to exercise slot reuse, and
reload checkpoint weights before repeating. Six queued requests with completion
lengths 8/32/16/48/8/32 exceed four running slots and mix new prefills with decode.
Every GDN forward asserts FP32 recurrent state.

The reference teacher-forces the actual served continuation using the release
HF **cached decode** path. Selected-token logprob error must be <=0.08 and at least
95% of generated tokens must be within 0.01 logprob of the HF argmax. Separate
full-vocabulary probes at prompt lengths 1/3/511/3071 require maximum
KL(served || HF-cached) <=1e-3, including 32 cached steps for short prompts.
All checks repeat after disk weight reload. Full-sequence HF evaluation is retained
as a separate diagnostic, alongside HF's own cached-versus-full discrepancy.

This distinction matters: at 810m, BF16 HF cached-versus-full maximum KL is
0.00176434 (TV 0.0244253) at a one-token prefix, exactly reproducing the original
SGLang-versus-full discrepancy. FP32 output logits reduce it only to 0.00151127.
Changing only the output head does not remove backbone/kernel rounding differences.
Do not interpret matching HF cached inference as proof that training and rollout
logprobs agree. The complete RL path needs its own importance-ratio check.
`--fp32-logits` is an optional diagnostic; the default recipe keeps BF16 logits.
The release FLA fast path is required; a slow Torch fallback cannot qualify.

Reports include checkpoint/config provenance, startup time, batch decode tokens/s,
probability errors, and reload results. A passing matrix qualifies this serving
configuration only. Packed native learner updates, live training publication,
preemption recovery, and learning quality require separate end-to-end GPU checks.

CPU evidence: all 192 actual release tensor names/shapes are covered; unknown
layers and missing tensors fail early. See the companion experiment log for GPU results and immutable report locations.


## Rhea results (2026-10-27)

The BF16 reference-backbone fallback passed all five final-midtrain sizes on one
RTX A6000 48GB: 32 functional cases and eight dense probes per size, including
reload, slot reuse, queued requests, and prompts through3071 tokens.
Maximum selected-token logprob errors and warm batch throughput:

| Size | Result | Max logprob error | Warm output tokens/s |
| --- | --- | ---: | ---: |
| 275m | PASSED | 1.23978e-05 | 62.1–67.1 |
| 450m | PASSED | 3.00407e-05 | 42.2–45.0 |
| 810m | PASSED | 0.000198603 | 39.2–45.0 |
| 1.4b | PASSED | 0.000464261 | 28.9–32.0 |
| 2.7b | PASSED | 9.9659e-05 | 19.8–22.4 |

The columns above are size, result, maximum logprob error, and warm batch output
tokens/s including prefill. Greedy agreement was100%. Report root:
`/weka/oe-adapt-default/michaeln/open-instruct-hybrid/qualification/rhea-hf-bf16-8bb6da4a`.
The tested adapter is SHA256
`73e345ec258aded60a37bea5e93d70a5bc5f753199bc41b004c809a6ebf4df62`.
The companion recipe selects this fallback and explicitly disables FP32 logits.
The packed/RadixAttention path remains unqualified at all five scales.
End-to-end native learner/publication/resume validation remains pending.
