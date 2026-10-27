"""Check the real Core model factory against all released 275m tensor shapes."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from miles.backends.core_utils import hybrid_models

ROOT = Path(__file__).with_name("hybrid_release")


def test_native_release_layout_and_streamed_publication():
    hf = SimpleNamespace(**json.loads((ROOT / "hf_config.json").read_text()))
    options = SimpleNamespace(model_config=str(ROOT / "native_config.json"), attention_backend="torch")
    config = hybrid_models.model_config_from_hf(hf, options)
    model = config.build(init_device="meta")
    actual = dict(hybrid_models.iter_export_state(SimpleNamespace(model=model), hf))
    shapes = {name: list(tensor.shape) for name, tensor in actual.items()}
    assert shapes == json.loads((ROOT / "tensor_shapes.json").read_text())
    assert all(tensor.dtype == torch.bfloat16 for tensor in actual.values())
