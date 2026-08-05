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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.interpolate_shear_to_half_level_cells import (
    interpolate_shear_to_half_level_cells,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests


class TestInterpolateShearToHalfLevelCells(stencil_tests.StencilTest):
    PROGRAM = interpolate_shear_to_half_level_cells
    OUTPUTS = ("mech_prod",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        shear: np.ndarray,
        e_bln_c_s: np.ndarray,
        wgtfac_c: np.ndarray,
        horizontal_start: int,
        horizontal_end: int,
        vertical_start: int,
        vertical_end: int,
        **kwargs: Any,
    ) -> dict:
        c2e = connectivities[dims.C2EDim]  # (n_cells, 3)

        # Edge -> cell average with the bilinear weights, (n_cells, nlev) full levels
        shear_c = np.sum(np.expand_dims(e_bln_c_s, axis=-1) * shear[c2e], axis=1)

        # Full -> half level interpolation: half level k mixes full levels k and k - 1.
        # Fortran jk = 2..nlev (1-based) -> k = 1..nlev-1 (0-based); the top and
        # bottom half-level rows are not computed.
        mech_prod = np.zeros_like(wgtfac_c)
        mech_prod[:, 1:-1] = (
            wgtfac_c[:, 1:-1] * shear_c[:, 1:] + (1.0 - wgtfac_c[:, 1:-1]) * shear_c[:, :-1]
        )

        mech_prod_out = np.zeros_like(mech_prod)
        mech_prod_out[horizontal_start:horizontal_end, vertical_start:vertical_end] = mech_prod[
            horizontal_start:horizontal_end, vertical_start:vertical_end
        ]
        return dict(mech_prod=mech_prod_out)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        shear = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)
        e_bln_c_s = data_alloc.random_field(grid, dims.CellDim, dims.C2EDim, dtype=ta.wpfloat)
        wgtfac_c = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
        )
        mech_prod = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
        )

        # Fortran: interpolate_rate_of_strain_full2half_edge2cell runs on
        # rl_start = 3, rl_end = min_rlcell_int - 1.
        cell_domain = h_grid.domain(dims.CellDim)
        horizontal_start = grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3))
        horizontal_end = grid.end_index(cell_domain(h_grid.Zone.HALO))
        assert horizontal_start < horizontal_end

        return dict(
            shear=shear,
            e_bln_c_s=e_bln_c_s,
            wgtfac_c=wgtfac_c,
            mech_prod=mech_prod,
            # Fortran jk = 2..nlev (1-based) -> half levels k = 1..nlev-1 (0-based)
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=1,
            vertical_end=gtx.int32(grid.num_levels),
        )
