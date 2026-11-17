import json
from types import SimpleNamespace

import pytest
import safetensors.torch
import torch

from miles.backends.core_utils import quantized_export as export
from miles.backends.core_utils.publication import quantization, tensors
from tools.convert_core_hf_quantized import convert

PREFIX = "model.layers.1.mlp.experts.0."
FP8 = {"quant_method": "fp8", "activation_scheme": "dynamic", "ignored_layers": ["lm_head"]}
NVFP4 = {"quant_method": "modelopt", "quant_algo": "NVFP4", "exclude_modules": ["lm_head"]}


def state(value=1):
    return [
        (PREFIX + role + ".weight", torch.full((16, 32), value, dtype=torch.bfloat16))
        for role in ("up_proj", "gate_proj", "down_proj")
    ]


def test_fp8_repeated_changed_restored_and_high_precision_head():
    head = torch.randn(8, 16, dtype=torch.float32)
    receipts = []
    for value in (1, 1, 2, 1):
        inputs = state(value) + [("lm_head.weight", head)]
        result = dict(export.iter_quantized_state(inputs, FP8))
        assert result["lm_head.weight"] is head
        for name, original in inputs[:-1]:
            weight = result[name]
            scale = result[name.replace(".weight", ".weight_scale")]
            assert weight.dtype == torch.float8_e4m3fn
            assert scale.dtype == torch.float32
            torch.testing.assert_close(weight.float() * scale, original.float())
            assert torch.equal(original, torch.full_like(original, value))
        receipts.append({name: value.view(torch.uint8).clone() for name, value in result.items()})
    for name in receipts[0]:
        assert torch.equal(receipts[0][name], receipts[1][name])
        assert torch.equal(receipts[0][name], receipts[3][name])
    assert not torch.equal(
        receipts[0][PREFIX + "gate_proj.weight_scale"], receipts[2][PREFIX + "gate_proj.weight_scale"]
    )


def test_nvfp4_pairs_before_bucketing_and_preserves_scale_types(monkeypatch):
    calls = []

    def quantize(weight):
        return (
            torch.zeros((16, 16), dtype=torch.uint8),
            torch.ones((16, 2)).to(torch.float8_e4m3fn),
            torch.tensor(0.25),
        )

    def pair(first, second):
        calls.append((first.clone(), second.clone()))
        return quantize(first), quantize(second)

    monkeypatch.setattr(export.nvfp4, "nvfp4_quantize_1d_pair", pair)
    monkeypatch.setattr(export.nvfp4, "nvfp4_quantize_1d", quantize)
    result = dict(export.iter_quantized_state(state(), NVFP4))
    assert len(calls) == 1
    assert len(result) == 9
    assert result[PREFIX + "up_proj.weight"].dtype == torch.uint8
    assert result[PREFIX + "up_proj.weight_scale"].dtype == torch.float8_e4m3fn
    torch.testing.assert_close(result[PREFIX + "up_proj.weight_scale_2"], result[PREFIX + "gate_proj.weight_scale_2"])


@pytest.mark.parametrize("names", [("up_proj",), ("up_proj", "down_proj"), ("up_proj", "up_proj")])
def test_incomplete_interleaved_duplicate_pairs_fail(names):
    weights = [(PREFIX + name + ".weight", torch.ones(16, 32)) for name in names]
    with pytest.raises(ValueError, match="pair|Duplicate"):
        # Missing/interleaved pairs fail before reaching any CUDA quantizer.
        list(export.iter_quantized_state(weights, NVFP4))


@pytest.mark.parametrize(
    "config",
    [
        {"quant_method": "fp8", "activation_scheme": "static"},
        {**FP8, "weight_block_size": [128, 128]},
        {"quant_method": "mxfp8"},
    ],
)
def test_unsupported_formats_fail(config):
    with pytest.raises(ValueError):
        list(export.iter_quantized_state(state(), config))


def test_requires_exclusions_for_nonexpert_matrices():
    with pytest.raises(ValueError, match="Non-expert"):
        list(export.iter_quantized_state([("model.layers.1.self_attn.q_proj.weight", torch.ones(16, 16))], FP8))


def test_resolve_uses_effective_updated_checkpoint_and_rejects_online(monkeypatch):
    group = SimpleNamespace(worker_type="regular", overrides={"model_path": "serialized"})
    model = SimpleNamespace(update_weights=True, model_path="base", server_groups=[group])
    deployment = SimpleNamespace(models=[model])
    paths = []

    def descriptor(path):
        paths.append(path)
        return {"model_type": "olmo3moe", "quantization_config": FP8}, {}

    monkeypatch.setattr(quantization.PretrainedConfig, "get_config_dict", descriptor)
    args = SimpleNamespace(
        hf_checkpoint="trainer",
        olmo_core=SimpleNamespace(quantized_export=True, stream_moe_export=True, expert_publication="per_expert"),
    )
    assert quantization.resolve(args, deployment=deployment) == FP8
    assert paths == ["serialized"]
    group.overrides["quantization"] = "modelopt_fp4"
    with pytest.raises(ValueError, match="conflicts"):
        quantization.resolve(args, deployment=deployment)


def test_default_does_not_resolve_or_change_existing_runs():
    assert quantization.resolve(SimpleNamespace(olmo_core=SimpleNamespace())) is None


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_zero_weights_have_finite_positive_scale(dtype):
    result = dict(export.iter_quantized_state([(PREFIX + "down_proj.weight", torch.zeros(16, 32, dtype=dtype))], FP8))
    assert result[PREFIX + "down_proj.weight_scale"].item() > 0
    assert torch.isfinite(result[PREFIX + "down_proj.weight"].float()).all()


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float8_e4m3fn, torch.float32, torch.bfloat16])
def test_publication_copies_preserve_bytes_dtype_and_storage(dtype):
    source = torch.arange(32).to(dtype).reshape(4, 8).T
    owned = tensors.copy_tensor(source, quantized=True)
    assert owned.dtype == source.dtype
    assert owned.is_contiguous()
    assert owned.data_ptr() != source.data_ptr()
    assert tensors.nbytes(source, quantized=True) == owned.view(torch.uint8).numel()
    assert torch.equal(owned.view(torch.uint8), source.contiguous().view(torch.uint8))
    source.zero_()
    assert owned.float().sum() > 0


def test_unquantized_publication_retains_bf16_contract():
    source = torch.ones(8, dtype=torch.float32)
    assert tensors.copy_tensor(source, quantized=False).dtype == torch.bfloat16
    assert tensors.nbytes(source, quantized=False) == 16


def test_mixed_dtype_bucket_alignment_and_byte_limit():
    scale = torch.ones(1, dtype=torch.float32)
    assert tensors.requires_flush(27, scale, quantized=True, limit=100)
    assert not tensors.requires_flush(28, scale, quantized=True, limit=100)
    assert tensors.requires_flush(28, scale, quantized=True, limit=30)
    assert not tensors.requires_flush(0, scale, quantized=True, limit=1)


def test_serialized_checkpoint_matches_publication_and_keeps_input(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "fp8"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "olmo3moe"}))
    weights = dict(state(2))
    weights["lm_head.weight"] = torch.randn(8, 16)
    safetensors.torch.save_file(weights, source / "model.safetensors")
    convert(source, destination, fmt="fp8", device="cpu", shard_bytes=520)
    config = json.loads((destination / "config.json").read_text())
    index = json.loads((destination / "model.safetensors.index.json").read_text())
    restored = {}
    for filename in set(index["weight_map"].values()):
        restored.update(safetensors.torch.load_file(destination / filename))
    expected = dict(export.iter_quantized_state(sorted(weights.items()), config["quantization_config"]))
    assert restored.keys() == expected.keys()
    assert index["metadata"]["total_size"] == sum(t.nbytes for t in expected.values())
    for name in expected:
        assert torch.equal(restored[name].view(torch.uint8), expected[name].view(torch.uint8))
    assert "quantization_config" not in json.loads((source / "config.json").read_text())
    with pytest.raises(FileExistsError):
        convert(source, destination, fmt="fp8", device="cpu")


def test_failed_conversion_never_exposes_partial_checkpoint(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "bad"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "olmo3moe"}))
    safetensors.torch.save_file({PREFIX + "up_proj.weight": torch.ones(16, 32)}, source / "model.safetensors")
    with pytest.raises(ValueError, match="Incomplete"):
        convert(source, destination, fmt="nvfp4", device="cpu")
    assert not destination.exists()
    assert not list(tmp_path.glob(".quantized-export-*"))
