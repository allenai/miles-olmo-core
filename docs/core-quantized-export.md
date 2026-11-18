# Core quantized weight export

This path automatically converts Core's ordinary per-expert HF export on the sender.
It reuses MILES' existing NVFP4 quantizer unchanged. A small Core-local FP8
helper follows Megatron's per-tensor recipe without changing Megatron files.
Parity tests compare against the checkout's Megatron implementation; Core clamps
scales in FP32 to additionally handle zero FP16 weights safely. Core itself does not currently expose a quantized HF
export option. Training weights and native resumable checkpoints stay unchanged.

## Prepare the initial rollout checkpoint

Run on a CUDA device for NVFP4; FP8 also supports CPU conversion:

```bash
python tools/convert_core_hf_quantized.py \
  --model-dir /models/core-bf16-hf \
  --save-dir /models/core-nvfp4-hf \
  --format nvfp4 --device cuda
```

Use `--format fp8` for FP8 E4M3, dynamic activations and per-tensor weight
scales. Both formats quantize routed experts only. Other weights retain their
source dtype, including an FP32 head when present. NVFP4 gate/up pairs share
one global scale. Conversion uses the same iterator as subsequent publication,
writes bounded CPU shards, refuses an existing destination and only exposes
the destination after successful conversion. KV-cache precision is configured
separately on the serving engine.

## Connect training and rollout

Keep the original BF16 checkpoint as `hf_checkpoint`. In the MILES SGLang
deployment config, give the updated actor model the converted `model_path`
and explicitly set `update_weights: true` (different model paths otherwise
default to frozen). Do not enable online quantization or application installers.
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
  block FP8, selective expert exclusion and nonliteral exclusion rules fail.
- The input checkpoint must be an ordinary unquantized HF safetensors export.
- Initial conversion and refresh must use the same NVTE quantizer environment
  (including any NVFP4 4-over-6 settings).
- CPU tests cover FP8 checkpoint/update byte parity, immutable publication
  copies, mixed dtypes, gate/up grouping and invalid configuration. The CPU
  NVFP4 grouping test uses a stub quantizer; it does not validate CUDA numerics.
- Serialized SGLang loading and repeated GPU refresh still require end-to-end
  qualification. Earlier online receiver-quantization tests do not qualify this
  path. Do not switch active runs based on CPU tests alone.
