"""Non-writing ranks consume export collectives without staging a full CPU model."""

from unittest import mock

import pytest
import torch

pytest.importorskip("olmo_core")

from miles.backends.core_utils import models


def test_nonwriter_consumes_all_exports_without_cpu_copies(monkeypatch):
    consumed = []
    tensors = [mock.Mock(), mock.Mock(), mock.Mock()]

    def exports(module, hf):
        for index, value in enumerate(tensors):
            consumed.append(index)
            yield str(index), value

    monkeypatch.setattr(models, "iter_export_state", exports)
    assert models.export_state(object(), object(), retain=False) == {}
    assert consumed == [0, 1, 2]
    for value in tensors:
        value.detach.assert_not_called()
        value.cpu.assert_not_called()


@pytest.mark.parametrize("options", [{}, {"retain": True}])
def test_writer_preserves_export_names_values_and_dtypes(monkeypatch, options):
    expected = {
        "weight": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4).T,
        "bias": torch.arange(3, dtype=torch.float32),
    }
    monkeypatch.setattr(models, "iter_export_state", lambda *args: iter(expected.items()))
    actual = models.export_state(object(), object(), **options)
    assert actual.keys() == expected.keys()
    for key, value in actual.items():
        assert value.device.type == "cpu"
        assert value.dtype == expected[key].dtype
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
