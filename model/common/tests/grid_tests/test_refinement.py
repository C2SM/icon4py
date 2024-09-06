# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import icon4py.model.common.dimension as dims
import icon4py.model.common.grid.refinement as refin

from . import utils


def out_of_range(dim: dims.Dimension):
    lower = range(-25, -9) if dim == dims.EdgeDim else range(-25, -5)
    for v in lower:
        yield v

    for v in range(15, 25):
        yield v


def refinement_value(dim: dims.Dimension):
    lower = -8 if dim == dims.EdgeDim else -4
    for v in range(lower, 14):
        yield v


# TODO (@halungge) fix this test -- too complex
@pytest.mark.parametrize("dim", utils.horizontal_dim())
def test_ordered(dim):
    for value in refinement_value(dim):
        ordered = refin.RefinementValue(dim, value)
        if dim == dims.EdgeDim:
            if ordered.value == 0 or ordered.value == -8:
                assert not ordered.is_ordered()
            else:
                assert ordered.is_ordered()
        else:
            if ordered.value == 0 or ordered.value == -4:
                assert not ordered.is_ordered()
            else:
                assert ordered.is_ordered()


@pytest.mark.parametrize("dim", utils.horizontal_dim())
def test_nested(dim):
    for value in refinement_value(dim):
        nested = refin.RefinementValue(dim, value)
        if nested.value < 0:
            assert nested.is_nested()
        else:
            assert not nested.is_nested()


@pytest.mark.parametrize("dim", utils.horizontal_dim())
def test_valid_refinement_values(dim):
    for value in out_of_range(dim):
        with pytest.raises(AssertionError):
            refin.RefinementValue(dim, value)
