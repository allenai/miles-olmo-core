"""Consecutive packing preserves ordering and token budgets."""

import pytest
from miles.backends.core_utils import packing


def test_plan_and_split_preserve_order_budget_and_membership():
    plans = [packing.plan([5, 8, 6, 7], 16), packing.plan([15, 15, 2, 2], 16)]
    assert list(map(len, plans)) == [2, 3]
    for plan in plans:
        equal = packing.equalize(plan, 3)
        assert len(equal) == 3 and all(equal)
        assert sum(equal, []) == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        packing.plan([17], 16)
    with pytest.raises(ValueError):
        packing.equalize(plans[0], 5)
