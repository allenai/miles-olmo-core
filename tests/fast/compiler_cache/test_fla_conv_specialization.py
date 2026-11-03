"""The convolution adapter must preserve runtime arithmetic and tuning choices."""

import ast
from types import SimpleNamespace
from unittest import mock

import pytest

from miles.backends.core_utils import fla_conv_compat


def test_only_unused_nb_annotation_changes():
    source = "def kernel(x, NB: tl.constexpr, D: tl.constexpr):\n    return x + D\n"
    result = fla_conv_compat.runtime_nb_source(source)
    before, after = ast.parse(source).body[0], ast.parse(result).body[0]
    assert ast.dump(ast.Module(body=before.body, type_ignores=[])) == ast.dump(
        ast.Module(body=after.body, type_ignores=[])
    )
    assert after.args.args[1].annotation is None
    assert ast.dump(before.args.args[2]) == ast.dump(after.args.args[2])


@pytest.mark.parametrize("expression", ["x + NB", "helper(NB)", "NB * D"])
def test_kernel_body_using_nb_is_rejected(expression):
    with pytest.raises(ValueError, match="used in the kernel body"):
        fla_conv_compat.runtime_nb_source(
            f"def kernel(x, NB: tl.constexpr, D: tl.constexpr):\n    return {expression}\n"
        )


def test_changed_declaration_is_rejected():
    with pytest.raises(ValueError, match="declaration"):
        fla_conv_compat.runtime_nb_source("def kernel(x, NB):\n    return x\n")


@pytest.fixture
def adapter(monkeypatch):
    class JITFunction:
        def __init__(self):
            self.src = "def kernel(x, NB: tl.constexpr, D: tl.constexpr):\n    return x + D\n"

    tuners = [SimpleNamespace(fn=JITFunction(), keys=["NB"], cache={}) for _ in range(2)]
    kernels = SimpleNamespace(**dict(zip(fla_conv_compat._KERNELS, tuners, strict=True)))
    modules = {
        "triton": SimpleNamespace(runtime=SimpleNamespace(JITFunction=JITFunction)),
        "fla.modules.conv.triton.kernels": kernels,
    }
    monkeypatch.setattr(fla_conv_compat, "_PATCHED", False)
    monkeypatch.setattr(fla_conv_compat.metadata, "version", lambda name: "0.5.2")
    monkeypatch.setattr(fla_conv_compat, "import_module", modules.__getitem__)
    # Exercise the real source guard without constructing GPU JIT objects.
    monkeypatch.setattr(
        fla_conv_compat, "dynamic_nb_kernel", lambda original, triton: fla_conv_compat.runtime_nb_source(original.src)
    )
    return tuners


@pytest.mark.parametrize("reason", ["version", "used_tuner", "missing_key", "declaration"])
def test_unsupported_adapter_warns_and_leaves_both_kernels_unchanged(adapter, monkeypatch, reason):
    originals = [tuner.fn for tuner in adapter]
    if reason == "version":
        monkeypatch.setattr(fla_conv_compat.metadata, "version", lambda name: "0.6.0")
    elif reason == "used_tuner":
        adapter[1].cache["previous"] = "config"
    elif reason == "missing_key":
        adapter[1].keys = []
    else:
        adapter[1].fn.src = "def kernel(x, NB):\n    return x\n"
    with mock.patch.object(fla_conv_compat.logger, "warning") as warning:
        assert fla_conv_compat.install() is False
    warning.assert_called_once()
    assert [tuner.fn for tuner in adapter] == originals
    assert not fla_conv_compat._PATCHED


def test_install_keeps_body_use_guard_fatal_and_atomic(adapter):
    originals = [tuner.fn for tuner in adapter]
    adapter[1].fn.src = "def kernel(x, NB: tl.constexpr,):\n    return x + NB\n"
    with pytest.raises(ValueError, match="used in the kernel body"):
        fla_conv_compat.install()
    assert [tuner.fn for tuner in adapter] == originals
    assert not fla_conv_compat._PATCHED


def test_supported_adapter_installs_once(adapter):
    assert fla_conv_compat.install() is True
    assert all("NB," in tuner.fn for tuner in adapter)
    assert fla_conv_compat.install() is False
