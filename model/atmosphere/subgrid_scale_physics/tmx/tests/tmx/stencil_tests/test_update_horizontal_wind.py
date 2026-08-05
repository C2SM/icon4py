# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.update_horizontal_wind import (
    update_horizontal_wind,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests


class TestUpdateHorizontalWind(stencil_tests.StencilTest):
    PROGRAM = update_horizontal_wind
    OUTPUTS = ("new_u", "new_v")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        u: np.ndarray,
        v: np.ndarray,
        tend_u: np.ndarray,
        tend_v: np.ndarray,
        dtime: float,
        horizontal_start: int,
        horizontal_end: int,
        vertical_start: int,
        vertical_end: int,
        **kwargs: Any,
    ) -> dict:
        new_u = np.zeros_like(u)
        new_v = np.zeros_like(v)
        hs, he = horizontal_start, horizontal_end
        vs, ve = vertical_start, vertical_end
        new_u[hs:he, vs:ve] = u[hs:he, vs:ve] + tend_u[hs:he, vs:ve] * dtime
        new_v[hs:he, vs:ve] = v[hs:he, vs:ve] + tend_v[hs:he, vs:ve] * dtime
        return dict(new_u=new_u, new_v=new_v)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        u = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        v = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        tend_u = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        tend_v = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        new_u = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        new_v = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)

        # Fortran: tmx 'domain' cell bounds, rl_start = grf_bdywidth_c + 1,
        # rl_end = min_rlcell_int.
        cell_domain = h_grid.domain(dims.CellDim)
        horizontal_start = grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        horizontal_end = grid.end_index(cell_domain(h_grid.Zone.LOCAL))
        assert horizontal_start < horizontal_end

        return dict(
            u=u,
            v=v,
            tend_u=tend_u,
            tend_v=tend_v,
            new_u=new_u,
            new_v=new_v,
            dtime=ta.wpfloat(300.0),
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
