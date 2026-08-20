# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import numpy as np

import icon4py.model.common.type_alias as types
from icon4py.model.atmosphere.tracer_advection.stencils.compute_horizontal_tracer_flux_upwind import (
    compute_horizontal_tracer_flux_upwind,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.testing import stencil_tests


class TestComputeHorizontalTracerFluxUpwind(stencil_tests.StencilTest):
    PROGRAM = compute_horizontal_tracer_flux_upwind
    OUTPUTS = ("p_out_e",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        p_cc: np.ndarray,
        p_mass_flx_e: np.ndarray,
        p_vn: np.ndarray,
        **kwargs,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        e2c = connectivities[dims.E2CDim]
        p_out_e = np.where(p_vn > 0.0, p_cc[e2c][:, 0], p_cc[e2c][:, 1]) * p_mass_flx_e
        return dict(p_out_e=p_out_e)

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        p_cc = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=types.wpfloat)
        p_mass_flx_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=types.wpfloat)
        p_vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=types.wpfloat)
        p_out_e = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, dtype=types.wpfloat)
        return dict(
            p_cc=p_cc,
            p_mass_flx_e=p_mass_flx_e,
            p_vn=p_vn,
            p_out_e=p_out_e,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
