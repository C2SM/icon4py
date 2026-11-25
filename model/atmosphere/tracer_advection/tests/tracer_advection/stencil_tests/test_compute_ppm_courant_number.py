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

from icon4py.model.atmosphere.tracer_advection.stencils.compute_ppm import (
    compute_ppm_courant_number,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.testing import stencil_tests


@pytest.mark.uses_concat_where
class TestComputePpm4gpuCourantNumber(stencil_tests.StencilTest):
    PROGRAM = compute_ppm_courant_number
    OUTPUTS = ("z_cfl",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        p_mflx_contra_v: np.ndarray,
        p_dtime: float,
        **kwargs: Any,
    ) -> dict:
        # Each case transports 1.25 unit cell masses across one cell, so the
        # expected Courant number is exactly the transported mass.
        return dict(z_cfl=p_dtime * p_mflx_contra_v)

    @stencil_tests.input_data_fixture(
        params=[
            pytest.param((8, 9, 1, 1.25), id="lower_boundary"),
            pytest.param((2, 9, 1, -1.25), id="upper_boundary"),
        ]
    )
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper,
        grid: base.Grid,
        request: pytest.FixtureRequest,
    ) -> dict:
        k_half_index, nlev, start_lvl_plus1, transported_mass = request.param
        p_cellmass_now = np.ones((grid.num_cells, grid.num_levels), dtype=ta.wpfloat)
        mflx_contra_v = np.zeros((grid.num_cells, grid.num_levels + 1), dtype=ta.wpfloat)
        mflx_contra_v[:, k_half_index] = transported_mass

        return dict(
            p_mflx_contra_v=data_alloc.as_field(mflx_contra_v, dims.CellDim, dims.KHalfDim),
            p_cellmass_now=data_alloc.as_field(p_cellmass_now, dims.CellDim, dims.KDim),
            z_cfl=data_alloc.zero_field(dims.CellDim, dims.KHalfDim),
            k_half=data_alloc.index_field(dims.KHalfDim),
            start_lvl_plus1=gtx.int32(start_lvl_plus1),
            nlev=gtx.int32(nlev),
            wp_eps=ta.wpfloat(1e-9),
            p_dtime=ta.wpfloat(1.0),
            horizontal_start=gtx.int32(0),
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=gtx.int32(1),
            vertical_end=gtx.int32(nlev + 1),
        )
