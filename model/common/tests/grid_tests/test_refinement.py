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
    
    lower = range(-36, refin._UNORDERED[dim][1]) 
    for v in lower:
        yield v

    upper = range(refin._MAX_ORDERED[dim] + 1, 36) 
    for v in upper:
        yield v


def refinement_value(dim: dims.Dimension):
    lower = refin._UNORDERED[dim][1] 
    upper = refin._MAX_ORDERED[dim]
    for v in range(lower, upper):
        yield v



@pytest.mark.parametrize("dim", utils.horizontal_dim())
def test_ordered(dim):
    for value in refinement_value(dim):
        ordered = refin.RefinementValue(dim, value)
        
        if ordered.value in refin._UNORDERED[dim]:
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
