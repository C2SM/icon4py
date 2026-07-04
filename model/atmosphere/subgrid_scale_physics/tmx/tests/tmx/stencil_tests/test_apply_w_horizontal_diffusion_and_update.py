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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.apply_w_horizontal_diffusion_and_update import (
    apply_w_horizontal_diffusion_and_update,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests


class TestApplyWHorizontalDiffusionAndUpdate(stencil_tests.StencilTest):
    """
    Fortran (mo_vdf.f90, 'Compute_diffusion_vert_wind') updates tend and w on
    half-level rows jk = 2..nlev (1-based) -> 0-based rows 1..nlev-1 (w = 0 at
    the top and bottom half levels); outside those rows tend keeps its input
    values and new_w stays zero ('CALL init(new_state)').
    """

    PROGRAM = apply_w_horizontal_diffusion_and_update
    OUTPUTS = ("new_w", "tend")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        hori_tend_e: np.ndarray,
        e_bln_c_s: np.ndarray,
        inv_rho_ic: np.ndarray,
        w: np.ndarray,
        tend: np.ndarray,
        dtime: float,
        horizontal_start: int,
        horizontal_end: int,
        vertical_start: int,
        vertical_end: int,
        **kwargs: Any,
    ) -> dict:
        c2e = connectivities[dims.C2EDim]  # (n_cells, 3)
        hori_tend_c = inv_rho_ic * np.sum(e_bln_c_s[:, :, np.newaxis] * hori_tend_e[c2e], axis=1)

        hs, he = horizontal_start, horizontal_end
        vs, ve = vertical_start, vertical_end
        tend_out = tend.copy()
        tend_out[hs:he, vs:ve] = tend[hs:he, vs:ve] + hori_tend_c[hs:he, vs:ve]
        new_w = np.zeros_like(w)
        new_w[hs:he, vs:ve] = w[hs:he, vs:ve] + tend_out[hs:he, vs:ve] * dtime
        return dict(new_w=new_w, tend=tend_out)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        hori_tend_e = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
        )
        e_bln_c_s = data_alloc.random_field(
            grid, dims.CellDim, dims.C2EDim, low=0.0, high=1.0, dtype=ta.wpfloat
        )
        inv_rho_ic = data_alloc.random_field(
            grid,
            dims.CellDim,
            dims.KDim,
            low=0.5,
            high=2.0,
            extend={dims.KDim: 1},
            dtype=ta.wpfloat,
        )
        w = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
        )
        tend = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
        )
        new_w = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
        )

        # Fortran: tmx 'domain' cell bounds, rl_start = grf_bdywidth_c + 1,
        # rl_end = min_rlcell_int.
        cell_domain = h_grid.domain(dims.CellDim)
        horizontal_start = grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        horizontal_end = grid.end_index(cell_domain(h_grid.Zone.LOCAL))
        assert horizontal_start < horizontal_end

        return dict(
            hori_tend_e=hori_tend_e,
            e_bln_c_s=e_bln_c_s,
            inv_rho_ic=inv_rho_ic,
            w=w,
            new_w=new_w,
            tend=tend,
            dtime=ta.wpfloat(300.0),
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            # Fortran jk = 2..nlev (1-based half levels) -> rows 1..nlev-1
            vertical_start=1,
            vertical_end=gtx.int32(grid.num_levels),
        )
