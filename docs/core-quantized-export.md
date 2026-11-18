# Core quantized weight export

This path automatically converts Core's ordinary per-expert HF export on the sender.
It reuses MILES' existing NVFP4 quantizer unchanged. FP8 uses the existing MILES CUDA `fp8_blockwise_cast` with 128x128 blocks,
as supported by Megatron, without changing Megatron files. A CPU reference
supports checkpoint conversion without CUDA. Both use FP32 scales and epsilon 1e-10. Core itself does not currently expose a quantized HF
export option. Training weights and native resumable checkpoints stay unchanged.

## Prepare the initial rollout checkpoint

Run on a CUDA device for NVFP4; FP8 also supports CPU conversion:

```bash
python tools/convert_core_hf_quantized.py \
  --model-dir /models/core-bf16-hf \
  --save-dir /models/core-nvfp4-hf \
  --format nvfp4 --device cuda
```

Use `--format fp8` for FP8 E4M3, dynamic activations and 128x128 block weight
scales (`weight_scale_inv`, FP32). Both formats quantize routed experts only. Other weights retain their
source dtype, including an FP32 head when present. NVFP4 gate/up pairs share
one global scale. Conversion uses the same iterator as subsequent publication,
writes bounded CPU shards, refuses an existing destination and only exposes
the destination after successful conversion. KV-cache precision is configured
separately on the serving engine.

## Connect training and rollout

Keep the original BF16 checkpoint as `hf_checkpoint`. In the MILES SGLang
deployment config, give the updated actor model the converted `model_path`
and explicitly set `update_weights: true` (different model paths otherwise
default to frozen). Set `sglang_moe_runner_backend=triton` for FP8. Other runner layouts are
currently rejected. Do not enable online quantization or application installers.
Allow SGLang to infer quantization from the serialized checkpoint configuration.

Export precision is inferred from the updated rollout checkpoints; no export flag
is required. Quantized checkpoints require these publication settings:

```toml
[core]
stream_moe_export = true
expert_publication = "per_expert"
weight_sync_mode = "flattened"
```

The actor resolves the actual updated engines' checkpoint configs once at
startup. BF16 checkpoints retain BF16 publication, including when KV-cache
quantization is enabled. Weight-quantization overrides without a serialized
quantized checkpoint, heterogeneous recipes, and conflicting overrides are
rejected. Prepare a converted checkpoint before selecting FP8/NVFP4 weights. Both direct
publication and engine-drain snapshots preserve packed weights, scales, dtypes,
and actual byte counts. Ordinary HF evaluation exports remain unquantized.

## Current limits and qualification

- Olmo MoE, all routed experts, per-expert publication only. Fused publication,
  per-tensor FP8, selective expert exclusion and nonliteral exclusion rules fail.
- The input checkpoint must be an ordinary unquantized HF safetensors export.
- Initial conversion and refresh must use the same NVTE quantizer environment
  (including any NVFP4 4-over-6 settings).
- CPU tests cover FP8 checkpoint/update byte parity, immutable publication
  copies, mixed dtypes, gate/up grouping and invalid configuration. The CPU
  NVFP4 grouping test uses a stub quantizer; it does not validate CUDA numerics.
- Serialized SGLang loading and repeated GPU refresh still require end-to-end
  qualification. Earlier online receiver-quantization tests do not qualify this
  path. Do not switch active runs based on CPU tests alone.

## Block FP8 refresh qualification

Per-tensor FP8 is intentionally rejected: the pinned SGLang initial finalizer
collapses gate/up scales, while its restore hook does not rebuild that loading
layout. Block FP8 with the Triton MoE runner keeps the serialized scale layout.
Both expert matrix dimensions must be divisible by 128; no implicit padding,
UE8M0 conversion or alternative kernel backend is selected. KV precision is
independent of weight precision.

`tests/manual/test_core_block_fp8_refresh.py` is an explicit one-GPU test, not
an RL run. It compares the native CUDA quantizer with the CPU reference, checks
repeated native finalization and graph-visible storage, and runs a tiny Olmo MoE
through serialized loading and split flattened CUDA-IPC publications with BF16
and FP8 E4M3 KV. Unchanged, changed, repeated and restored states are compared
with initial/cold-load generation. The fixture uses one full-attention layer;
it does not qualify production-scale hybrid KDA behavior, NCCL, or TP>1.
Set `CORE_FP8_FIXTURE_BUILDER` to olmo-sglang's
`tools/create_tiny_parity_checkpoint.py`, `CORE_FP8_TEST_OUTPUT` to an empty
output directory, and run that file explicitly with pytest. GPU results remain
pending until an actual execution receipt is recorded.
