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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.modify_w_diffusion_matrix_boundary import (
    modify_w_diffusion_matrix_boundary,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests


class TestModifyWDiffusionMatrixBoundary(stencil_tests.StencilTest):
    """
    Fortran (mo_vdf.f90, 'Compute_diffusion_vert_wind', w = 0 boundary condition):
        b(2)    += 2 * km_c(1)    * inv_dzf(1)    * inv_mair_ic(2)
        b(nlev) += 2 * km_c(nlev) * inv_dzf(nlev) * inv_mair_ic(nlev)
    (1-based rows) -> 0-based rows 1 and nlev-1; all other rows of b untouched.
    """

    PROGRAM = modify_w_diffusion_matrix_boundary
    OUTPUTS = ("b",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        b: np.ndarray,
        km_c: np.ndarray,
        inv_dz: np.ndarray,
        inv_mair_ic: np.ndarray,
        horizontal_start: int,
        horizontal_end: int,
        vertical_start: int,
        vertical_end: int,
        **kwargs: Any,
    ) -> dict:
        b_out = b.copy()
        hs, he = horizontal_start, horizontal_end
        top = vertical_start
        bottom = vertical_end - 1
        b_out[hs:he, top] += (
            2.0 * km_c[hs:he, top - 1] * inv_dz[hs:he, top - 1] * inv_mair_ic[hs:he, top]
        )
        b_out[hs:he, bottom] += (
            2.0 * km_c[hs:he, bottom] * inv_dz[hs:he, bottom] * inv_mair_ic[hs:he, bottom]
        )
        return dict(b=b_out)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        b = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        km_c = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=0.0, high=1.0, dtype=ta.wpfloat
        )
        inv_dz = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=0.1, high=2.0, dtype=ta.wpfloat
        )
        inv_mair_ic = data_alloc.random_field(
            grid,
            dims.CellDim,
            dims.KDim,
            low=0.1,
            high=2.0,
            extend={dims.KDim: 1},
            dtype=ta.wpfloat,
        )

        # Fortran: tmx 'domain' cell bounds, rl_start = grf_bdywidth_c + 1,
        # rl_end = min_rlcell_int.
        cell_domain = h_grid.domain(dims.CellDim)
        horizontal_start = grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        horizontal_end = grid.end_index(cell_domain(h_grid.Zone.LOCAL))
        assert horizontal_start < horizontal_end

        return dict(
            b=b,
            km_c=km_c,
            inv_dz=inv_dz,
            inv_mair_ic=inv_mair_ic,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            # w solve bounds: minlvl = 2, maxlvl = nlev (1-based) -> (1, nlev)
            vertical_start=1,
            vertical_end=gtx.int32(grid.num_levels),
        )
