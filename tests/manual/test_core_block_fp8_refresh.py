"""Explicit one-GPU qualification; no model downloads or RL training.

Run with CORE_FP8_FIXTURE_BUILDER pointing to olmo-sglang's
 tools/create_tiny_parity_checkpoint.py and CORE_FP8_TEST_OUTPUT to a durable dir.
This exercises native SGLang, without application quantization installers.
"""

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import safetensors.torch
import torch

from miles.backends.core_utils import fp8_export, quantized_export
from miles.backends.core_utils.publication import tensors
from tools.convert_core_hf_quantized import convert

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="explicit CUDA qualification")


def _native_layer():
    from sglang.srt.layers.moe.utils import initialize_moe_config
    from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
    from sglang.srt.runtime_context import get_context
    from sglang.srt.server_args import ServerArgs

    get_context().set_server_args(ServerArgs(model_path="unused", moe_runner_backend="triton"))
    initialize_moe_config()
    method = Fp8MoEMethod(Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128]))
    with torch.device("cuda"):
        layer = torch.nn.Module()
        layer.moe_runner_config = SimpleNamespace(is_gated=True)
        layer.num_local_experts = 4
        layer.intermediate_size_per_partition = 256
        layer.hidden_size = 128
        layer.use_presharded_weights = False
        layer.moe_tp_size = 1
        method.create_weights(layer, 4, 128, 256, torch.bfloat16)
    return layer, method


def _load_layer(layer, sources, factor):
    for name, values in sources.items():
        for expert, value in enumerate(values):
            packed, scales = fp8_export.quantize_blockwise(value * factor)
            getattr(layer, name).data[expert].copy_(packed)
            getattr(layer, name + "_scale_inv").data[expert].copy_(scales)


def test_native_block_fp8_repeated_refresh_storage_and_cuda_parity():
    torch.manual_seed(17)
    layer, method = _native_layer()
    sources = {
        name: torch.randn_like(getattr(layer, name), dtype=torch.bfloat16) for name in ("w13_weight", "w2_weight")
    }
    # Exercise the actual shared CUDA quantizer, comparing with the CPU reference.
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        value = sources["w2_weight"][0].to(dtype)
        gpu = fp8_export.quantize_blockwise(value)
        cpu = fp8_export.quantize_blockwise(value.cpu())
        for actual, expected in zip(gpu, cpu, strict=True):
            torch.testing.assert_close(actual.float().cpu(), expected.float(), rtol=0, atol=0)
    _load_layer(layer, sources, 1)
    method.process_weights_after_loading(layer)
    serving = dict(layer.named_parameters())
    pointers = {name: value.data_ptr() for name, value in serving.items()}
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_sum = sum(value.float().sum() for value in serving.values())
    for factor in (1, -3, -3, 1):
        method.restore_weights_before_loading(layer)
        _load_layer(layer, sources, factor)
        method.process_weights_after_loading(layer)
        cold, cold_method = _native_layer()
        _load_layer(cold, sources, factor)
        cold_method.process_weights_after_loading(cold)
        for name, value in layer.named_parameters():
            assert value is serving[name]
            assert value.data_ptr() == pointers[name]
            torch.testing.assert_close(value.float(), getattr(cold, name).float(), rtol=0, atol=0)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(graph_sum, sum(value.float().sum() for value in layer.parameters()), rtol=0, atol=0)


def _fixture(root):
    spec = importlib.util.spec_from_file_location("tiny_olmo_fixture", os.environ["CORE_FP8_FIXTURE_BUILDER"])
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    original = builder._config

    def config(profile, tokenizer, max_position_embeddings):
        result = original(profile, tokenizer, max_position_embeddings)
        result.update(
            model_type="olmo3moe",
            hidden_size=256,
            attention_hidden_size=256,
            head_dim=128,
            num_attention_heads=2,
            num_key_value_heads=1,
            latent_moe_dim=128,
            moe_intermediate_size=256,
            shared_expert_intermediate_size=256,
            layer_types=["full_attention"],
            num_hidden_layers=1,
        )
        return result

    builder._config = config  # Test fixture dimensions only; no serving/runtime patch.
    source = root / "source"
    builder.build_checkpoint(source, profile="hybrid-moe", max_position_embeddings=512)
    weights = safetensors.torch.load_file(source / "model.safetensors")
    packed = root / "serialized"
    convert(source, packed, fmt="fp8", device="cuda", shard_bytes=131072)
    recipe = json.loads((packed / "config.json").read_text())["quantization_config"]
    return packed, weights, recipe


def _variant(weights, changed):
    return [
        (name, value * (-1 if changed and ".mlp.experts." in name else 1)) for name, value in sorted(weights.items())
    ]


def _publish(engine, weights, recipe, changed):
    from sglang.srt.utils import MultiprocessingSerializer
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

    def check(result):
        if isinstance(result, tuple):
            assert result[0], result
        elif isinstance(result, dict):
            assert result.get("success", True), result
        else:
            assert getattr(result, "success", True), result

    def flush(bucket):
        flat = FlattenedTensorBucket(named_tensors=bucket)
        payload = MultiprocessingSerializer.serialize(
            {"flattened_tensor": flat.get_flattened_tensor(), "metadata": flat.get_metadata()}
        )
        check(engine.update_weights_from_tensor([payload], load_format="flattened_bucket", flush_cache=False))

    check(engine.begin_weight_update())
    bucket, size = [], 0
    for name, value in quantized_export.iter_quantized_state(
        ((n, t.cuda()) for n, t in _variant(weights, changed)), recipe
    ):
        if tensors.requires_flush(size, value, quantized=True, limit=65536):
            flush(bucket)
            bucket, size = [], 0
        bucket.append((name, tensors.copy_tensor(value, quantized=True)))
        size += value.nbytes
    if bucket:
        flush(bucket)
    check(engine.end_weight_update())
    engine.flush_cache()


def _generate(engine):
    output = engine.generate(
        input_ids=[[14, 15, 16, 17], [22, 23, 24, 25]],
        sampling_params={"temperature": 0, "max_new_tokens": 8, "ignore_eos": True},
        return_logprob=True,
    )
    for item in output:
        assert len(item["output_ids"]) == 8
        assert all(torch.isfinite(torch.tensor(x[0])) for x in item["meta_info"]["output_token_logprobs"])
    return [(item["output_ids"], [x[0] for x in item["meta_info"]["output_token_logprobs"]]) for item in output]


@pytest.mark.parametrize("kv_dtype", ["bf16", "fp8_e4m3"])
def test_olmo_serialized_refresh_with_kv(kv_dtype):
    import olmo_sglang
    from sglang import Engine

    root = Path(os.environ["CORE_FP8_TEST_OUTPUT"]) / kv_dtype
    root.mkdir(parents=True, exist_ok=True)
    packed, weights, recipe = _fixture(root)
    olmo_sglang.register()
    options = dict(
        model_path=str(packed),
        trust_remote_code=True,
        dtype="bfloat16",
        tp_size=1,
        context_length=512,
        max_running_requests=2,
        max_total_tokens=4096,
        mem_fraction_static=0.3,
        disable_radix_cache=True,
        cuda_graph_backend_decode="full",
        cuda_graph_max_bs_decode=2,
        cuda_graph_backend_prefill="disabled",
        attention_backend="triton",
        moe_runner_backend="triton",
        kv_cache_dtype=kv_dtype,
        random_seed=17,
        enable_fp32_lm_head=True,
        log_level="warning",
    )
    engine = Engine(**options)
    receipts = []
    try:
        initial = _generate(engine)
        for changed in (False, True, True, False):
            _publish(engine, weights, recipe, changed)
            actual = _generate(engine)
            if not changed:
                for (ids, scores), (expected_ids, expected_scores) in zip(actual, initial, strict=True):
                    assert ids == expected_ids
                    torch.testing.assert_close(torch.tensor(scores), torch.tensor(expected_scores), rtol=0, atol=1e-5)
            receipts.append({"changed": changed, "outputs": actual})
    finally:
        engine.shutdown()
    # Cold-load the changed checkpoint after releasing the original engine.
    changed_source = root / "changed-source"
    changed_source.mkdir()
    for path in (root / "source").iterdir():
        if path.name != "model.safetensors":
            (changed_source / path.name).write_bytes(path.read_bytes())
    safetensors.torch.save_file(dict(_variant(weights, True)), changed_source / "model.safetensors")
    changed_path = root / "changed-serialized"
    convert(changed_source, changed_path, fmt="fp8", device="cuda")
    cold = Engine(**{**options, "model_path": str(changed_path)})
    try:
        expected = _generate(cold)
        for receipt in receipts:
            if receipt["changed"]:
                for (ids, scores), (expected_ids, expected_scores) in zip(receipt["outputs"], expected, strict=True):
                    assert ids == expected_ids
                    torch.testing.assert_close(torch.tensor(scores), torch.tensor(expected_scores), rtol=0, atol=1e-5)
    finally:
        cold.shutdown()
    (root / "qualified.json").write_text(json.dumps({"kv_cache_dtype": kv_dtype, "receipts": receipts}, indent=2))
