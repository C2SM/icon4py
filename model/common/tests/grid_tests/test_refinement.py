# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import icon4py.model.common.grid.refinement as refin


def out_of_range():
    for v in range(-15, -4):
        yield v
    for v in range(15, 20):
        yield v


def refinement_value():
    for v in range(-4, 14):
        yield v


@pytest.mark.parametrize("value", refinement_value())
def test_ordered(value):
    ordered = refin.RefinementValue(value)
    if ordered.value == 0 or ordered.value == -4:
        assert not ordered.is_ordered()
    else:
        assert ordered.is_ordered()


@pytest.mark.parametrize("value", refinement_value())
def test_nested(value):
    nested = refin.RefinementValue(value)
    if nested.value < 0:
        assert nested.is_nested()
    else:
        assert not nested.is_nested()


@pytest.mark.parametrize("value", out_of_range())
def test_valid_refinement_values(value):
    with pytest.raises(AssertionError):
        refin.RefinementValue(value)
