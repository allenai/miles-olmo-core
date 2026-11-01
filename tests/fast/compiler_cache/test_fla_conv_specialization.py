"""The convolution adapter must preserve runtime arithmetic and tuning choices."""

import ast

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
