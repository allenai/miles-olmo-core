"""The filtered fork must construct models without excluded plugin arguments."""

import sys
from unittest.mock import Mock

import pytest

pytest.importorskip("megatron")

from megatron.training import arguments as megatron_arguments
from tools import convert_hf_to_torch_dist as conversion

from miles.backends.megatron_utils import model_provider


@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
def test_conversion_parser_reaches_model_provider(monkeypatch, tmp_path, tensor_parallel_size):
    monkeypatch.setenv("WORLD_SIZE", str(tensor_parallel_size))
    monkeypatch.setenv("CONVERT_KEEP_PP1", "1")
    monkeypatch.setattr(megatron_arguments, "get_device_arch_version", lambda: 10)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_hf_to_torch_dist.py",
            "--hf-checkpoint",
            str(tmp_path),
            "--save",
            str(tmp_path / "converted"),
            "--num-layers",
            "2",
            "--hidden-size",
            "16",
            "--num-attention-heads",
            "2",
            "--seq-length",
            "16",
            "--vocab-size",
            "128",
            "--tensor-model-parallel-size",
            str(tensor_parallel_size),
            "--transformer-impl",
            "local",
        ],
    )
    args = conversion.get_args()
    assert args.tensor_model_parallel_size == tensor_parallel_size
    # Run the real parser, config construction, layer-spec selection and provider.
    # Only model allocation is replaced: it needs CUDA and initialized process groups.
    constructor = Mock(return_value=object())
    monkeypatch.setattr(model_provider, "GPTModel", constructor)
    model = model_provider.get_model_provider_func(args)()
    assert model is constructor.return_value
    config = constructor.call_args.kwargs["config"]
    assert config.hidden_size == 16
    assert config.tensor_model_parallel_size == tensor_parallel_size
