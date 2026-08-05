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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.compute_w_vertical_diffusion_rhs import (
    compute_w_vertical_diffusion_rhs,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests


class TestComputeWVerticalDiffusionRhs(stencil_tests.StencilTest):
    """
    Fortran (mo_vdf.f90, 'Compute_diffusion_vert_wind') computes the w-solve rhs on
    half-level rows jk = 2..nlev (1-based), i.e. rows 1..nlev-1 with 0-based
    indexing; rows 0 and nlev of the half-level outputs stay untouched.
    """

    PROGRAM = compute_w_vertical_diffusion_rhs
    OUTPUTS = ("rhs", "inv_rho_ic", "inv_mair_ic")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        rho_ic: np.ndarray,
        inv_ddqz_z_half: np.ndarray,
        km_c: np.ndarray,
        div_c: np.ndarray,
        horizontal_start: int,
        horizontal_end: int,
        vertical_start: int,
        vertical_end: int,
        **kwargs: Any,
    ) -> dict:
        z_1by3 = 1.0 / 3.0

        inv_rho_ic = 1.0 / rho_ic  # (n_cells, nlev + 1)
        inv_mair_ic = inv_rho_ic * inv_ddqz_z_half

        # rhs(k) = 2 * inv_mair_ic(k) * (km_c(k) * div_c(k) - km_c(k-1) * div_c(k-1)) / 3
        # on half rows k = 1..nlev-1, reading the full levels below (k) and above (k-1).
        rhs = np.zeros_like(rho_ic)
        rhs[:, 1:-1] = (
            2.0
            * inv_mair_ic[:, 1:-1]
            * (km_c[:, 1:] * z_1by3 * div_c[:, 1:] - km_c[:, :-1] * z_1by3 * div_c[:, :-1])
        )

        hs, he = horizontal_start, horizontal_end
        vs, ve = vertical_start, vertical_end
        rhs_out = np.zeros_like(rhs)
        inv_rho_ic_out = np.zeros_like(inv_rho_ic)
        inv_mair_ic_out = np.zeros_like(inv_mair_ic)
        rhs_out[hs:he, vs:ve] = rhs[hs:he, vs:ve]
        inv_rho_ic_out[hs:he, vs:ve] = inv_rho_ic[hs:he, vs:ve]
        inv_mair_ic_out[hs:he, vs:ve] = inv_mair_ic[hs:he, vs:ve]
        return dict(rhs=rhs_out, inv_rho_ic=inv_rho_ic_out, inv_mair_ic=inv_mair_ic_out)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        rho_ic = data_alloc.random_field(
            grid,
            dims.CellDim,
            dims.KDim,
            low=0.5,
            high=2.0,
            extend={dims.KDim: 1},
            dtype=ta.wpfloat,
        )
        inv_ddqz_z_half = data_alloc.random_field(
            grid,
            dims.CellDim,
            dims.KDim,
            low=0.1,
            high=2.0,
            extend={dims.KDim: 1},
            dtype=ta.wpfloat,
        )
        km_c = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=0.0, high=1.0, dtype=ta.wpfloat
        )
        div_c = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)

        rhs = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
        )
        inv_rho_ic = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
        )
        inv_mair_ic = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
        )

        # Fortran: tmx 'domain' cell bounds, rl_start = grf_bdywidth_c + 1,
        # rl_end = min_rlcell_int.
        cell_domain = h_grid.domain(dims.CellDim)
        horizontal_start = grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        horizontal_end = grid.end_index(cell_domain(h_grid.Zone.LOCAL))
        assert horizontal_start < horizontal_end

        return dict(
            rho_ic=rho_ic,
            inv_ddqz_z_half=inv_ddqz_z_half,
            km_c=km_c,
            div_c=div_c,
            rhs=rhs,
            inv_rho_ic=inv_rho_ic,
            inv_mair_ic=inv_mair_ic,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            # Fortran jk = 2..nlev (1-based half levels) -> rows 1..nlev-1
            vertical_start=1,
            vertical_end=gtx.int32(grid.num_levels),
        )
