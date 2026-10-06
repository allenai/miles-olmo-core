"""Bind adapter arguments against installed Core without allocating CUDA state."""

import inspect
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("olmo_core")

from olmo_core.train.train_module import transformer as train_transformer

from miles.backends.core_utils import moe_models


@pytest.mark.parametrize("clip_grad", [0.5, 1.0])
def test_moe_clipping_is_optimizer_owned(clip_grad, monkeypatch):
    signature = inspect.signature(train_transformer.OLMoDDPTrainModule.__init__)
    captured = {}

    def constructor(self, **kwargs):
        signature.bind(self, **kwargs)
        captured.update(kwargs)

    monkeypatch.setattr(train_transformer.OLMoDDPTrainModule, "__init__", constructor)
    common = dict(
        model=object(),
        rank_microbatch_size=16,
        max_sequence_length=16,
        compile_model=False,
        device=torch.device("cpu"),
        max_grad_norm=clip_grad,
    )
    original = common.copy()
    args = SimpleNamespace(
        clip_grad=clip_grad,
        olmo_core=SimpleNamespace(
            compile_optimizer=False,
            use_reduce_scatter=False,
            expert_parallel_size=1,
            checkpoint_save_options=lambda: {},
        ),
    )
    moe_models.build_train_module(args, common=common, optim={"lr": 1e-6}, hf_config=None, hf_state={})
    assert "max_grad_norm" not in captured
    assert captured["optim"].max_grad_norm == clip_grad
    assert common == original
    # Dense Core still accepts the shared keyword; this repair must not remove it there.
    dense_signature = inspect.signature(train_transformer.TransformerTrainModule.__init__)
    dense_signature.bind_partial(None, **common)
