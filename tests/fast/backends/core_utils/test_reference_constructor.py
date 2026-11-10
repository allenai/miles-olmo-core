"""Reference construction must select Core's evaluation mode without training buffers."""

import inspect
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("olmo_core")

from olmo_core.train.train_module import transformer as train_transformer

from miles.backends.core_utils import models


@pytest.mark.parametrize("reference", [False, True])
@pytest.mark.parametrize("moe", [False, True])
def test_reference_constructor(reference, moe, monkeypatch):
    config_type = models.transformer_config.OLMoDDPModelConfig if moe else models.transformer.TransformerConfig
    config = object.__new__(config_type)
    model = torch.nn.Linear(4, 4)
    hf = SimpleNamespace(model_type="olmo3moe" if moe else "olmo3")
    captured = {}
    loaded = []
    module_type = train_transformer.OLMoDDPTrainModule if moe else train_transformer.TransformerTrainModule
    signature = inspect.signature(module_type.__init__)

    def constructor(self, **kwargs):
        signature.bind(self, **kwargs)
        captured.update(kwargs)
        if moe:
            self._pp_config = None
            self.model_parts = [kwargs["model"]]
        else:
            self.model = kwargs["model"]
        self.optim = None if kwargs.get("eval_only", False) else object()

    def load_hf(source, **kwargs):
        loaded.append(source)
        return torch.nn.Linear(4, 4)

    monkeypatch.setattr(module_type, "__init__", constructor)
    monkeypatch.setattr(config_type, "build", lambda self, **kwargs: model)
    monkeypatch.setattr(models, "model_config_from_hf", lambda *args: config)
    monkeypatch.setattr(models, "load_hf_config", lambda *args, **kwargs: hf)
    monkeypatch.setattr(models.transformers.PretrainedConfig, "get_config_dict", lambda *args: ({}, {}))
    monkeypatch.setattr(models.transformers.AutoModelForCausalLM, "from_pretrained", load_hf)
    monkeypatch.setattr(models.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(models, "startup_stage", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(models.router_objective, "install", lambda *args: None)
    backend = models._backend("moe" if moe else "standard")
    monkeypatch.setattr(backend, "validate_training_options", lambda *args: None)
    if moe:
        monkeypatch.setattr(backend, "prepare_model_config", lambda *args: None)
    else:
        monkeypatch.setattr(backend.dist, "get_world_size", lambda: 1)
        monkeypatch.setattr(backend.convert, "convert_state_from_hf", lambda *args, **kwargs: model.state_dict())
    options = SimpleNamespace(
        activation_checkpointing=False,
        publication_mode="barrier",
        packing_max_tokens=16,
        max_sequence_length=16,
        compile_model=False,
        compile_optimizer=False,
        use_reduce_scatter=False,
        expert_parallel_size=1,
        checkpoint_save_options=lambda: {},
    )
    args = SimpleNamespace(
        hf_checkpoint="policy",
        seed=17,
        olmo_core=options,
        clip_grad=1.0,
        lr=1e-6,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        weight_decay=0.0,
    )
    module, _, _ = models.build_train_module(args, source="reference" if reference else None, reference=reference)
    assert loaded == ["reference" if reference else "policy"]
    assert all(p.requires_grad is not reference for p in module.model.parameters())
    assert captured.get("eval_only", False) is (reference and moe)
    assert (module.optim is None) is (reference and moe)
