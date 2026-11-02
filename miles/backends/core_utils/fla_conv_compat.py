"""Remove an unused FLA convolution specialization without changing autotuning."""

import ast
from importlib import import_module, metadata
from types import FunctionType

from miles.utils import logger_utils

logger = logger_utils.setup_logger(__name__)
_PATCHED = False
_KERNELS = ("causal_conv1d_fwd_kernel", "causal_conv1d_bwd_kernel")


class UnsupportedDeclaration(ValueError):
    """The optional adapter does not recognize this kernel signature."""


def runtime_nb_source(source):
    """Fail closed if the pinned unused bucket ever starts affecting arithmetic."""
    tree = ast.parse(source)
    if any(isinstance(node, ast.Name) and node.id == "NB" for node in ast.walk(tree)):
        raise ValueError("FLA convolution NB is used in the kernel body; dynamic adaptation is not qualified")
    if source.count("NB: tl.constexpr,") != 1:
        raise UnsupportedDeclaration("Unexpected FLA convolution NB declaration")
    return source.replace("NB: tl.constexpr,", "NB,")


def dynamic_nb_kernel(original, triton):
    source = runtime_nb_source(original.src)
    fn = original.fn
    clone = FunctionType(fn.__code__, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__)
    clone.__annotations__ = {key: value for key, value in fn.__annotations__.items() if key != "NB"}
    clone.__module__ = fn.__module__
    clone.__qualname__ = fn.__qualname__
    clone.__kwdefaults__ = fn.__kwdefaults__
    updated = triton.jit(clone, do_not_specialize=[*original.do_not_specialize, "NB"])
    updated._unsafe_update_src(source)
    return updated


def install():
    """Install before the first convolution; preserve every autotuning key/config.

    NB remains a runtime argument and an autotuning key. It cannot affect emitted
    arithmetic because the source guard requires it to be unused. Native Triton
    keys distinguish this new signature from previously compiled kernels.
    """
    global _PATCHED
    if _PATCHED:
        return False
    if metadata.version("flash-linear-attention") != "0.5.2":
        logger.warning("Skipping convolution specialization adapter: requires flash-linear-attention==0.5.2")
        return False
    triton = import_module("triton")
    kernels = import_module("fla.modules.conv.triton.kernels")
    replacements = []
    for name in _KERNELS:
        tuner = getattr(kernels, name)
        while not isinstance(tuner.fn, triton.runtime.JITFunction):
            tuner = tuner.fn
        if "NB" not in tuner.keys or tuner.cache:
            logger.warning("Skipping convolution specialization adapter: %s requires an unused NB autotuner", name)
            return False
        try:
            replacements.append((tuner, dynamic_nb_kernel(tuner.fn, triton)))
        except UnsupportedDeclaration as error:
            logger.warning("Skipping convolution specialization adapter for %s: %s", name, error)
            return False
    for tuner, updated in replacements:
        tuner.fn = updated
    _PATCHED = True
    logger.info("FLA convolution: NB autotuning buckets retained; unused compiler specialization disabled")
    return True
