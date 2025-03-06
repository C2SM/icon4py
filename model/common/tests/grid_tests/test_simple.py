# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import horizontal as h_grid, simple


def domain_generator():
    for dim in (dims.EdgeDim, dims.CellDim, dims.VertexDim):
        for z in h_grid.Zone:
            try:
                domain = h_grid.domain(dim)(z)
                yield domain
            except AssertionError:
                pass


@pytest.mark.parametrize("domain", domain_generator())
def test_start_index(domain):
    simple_grid = simple.SimpleGrid()
    if domain.zone in (h_grid.Zone.HALO, h_grid.Zone.HALO_LEVEL_2):
        assert simple_grid.start_index(domain) == simple_grid.size[domain.dim]
    else:
        assert simple_grid.start_index(domain) == 0


@pytest.mark.parametrize("domain", domain_generator())
def test_end_index(domain):
    simple_grid = simple.SimpleGrid()
    assert simple_grid.end_index(domain) == simple_grid.size[domain.dim]


def test_has_skip_values():
    simple_grid = simple.SimpleGrid()
    assert not simple_grid.has_skip_values()
