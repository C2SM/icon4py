# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import pytest

import icon4py.model.common.test_utils.helpers as helpers
from icon4py.model.atmosphere.advection.stencils.compute_positive_definite_horizontal_multiplicative_flux_factor import (
    compute_positive_definite_horizontal_multiplicative_flux_factor,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.settings import xp


class TestComputePositiveDefiniteHorizontalMultiplicativeFluxFactor(helpers.StencilTest):
    PROGRAM = compute_positive_definite_horizontal_multiplicative_flux_factor
    OUTPUTS = ("r_m",)

    @staticmethod
    def reference(
        grid,
        geofac_div: xp.ndarray,
        p_cc: xp.ndarray,
        p_rhodz_now: xp.ndarray,
        p_mflx_tracer_h: xp.ndarray,
        p_dtime,
        dbl_eps,
        **kwargs,
    ) -> dict:
        geofac_div = helpers.reshape(geofac_div, grid.connectivities[dims.C2EDim].shape)
        geofac_div = xp.expand_dims(xp.asarray(geofac_div), axis=-1)
        p_m_0 = xp.maximum(
            0.0,
            p_mflx_tracer_h[grid.connectivities[dims.C2EDim][:, 0]] * geofac_div[:, 0] * p_dtime,
        )
        p_m_1 = xp.maximum(
            0.0,
            p_mflx_tracer_h[grid.connectivities[dims.C2EDim][:, 1]] * geofac_div[:, 1] * p_dtime,
        )
        p_m_2 = xp.maximum(
            0.0,
            p_mflx_tracer_h[grid.connectivities[dims.C2EDim][:, 2]] * geofac_div[:, 2] * p_dtime,
        )

        p_m = p_m_0 + p_m_1 + p_m_2
        r_m = xp.minimum(1.0, p_cc * p_rhodz_now / (p_m + dbl_eps))

        return dict(r_m=r_m)

    @pytest.fixture
    def input_data(self, grid) -> dict:
        geofac_div = helpers.random_field(grid, dims.CellDim, dims.C2EDim)
        geofac_div_new = helpers.as_1D_sparse_field(geofac_div, dims.CEDim)
        p_cc = helpers.random_field(grid, dims.CellDim, dims.KDim)
        p_rhodz_now = helpers.random_field(grid, dims.CellDim, dims.KDim)
        p_mflx_tracer_h = helpers.random_field(grid, dims.EdgeDim, dims.KDim)
        r_m = helpers.zero_field(grid, dims.CellDim, dims.KDim)
        p_dtime = xp.float64(5)
        dbl_eps = xp.float64(1e-9)
        return dict(
            geofac_div=geofac_div_new,
            p_cc=p_cc,
            p_rhodz_now=p_rhodz_now,
            p_mflx_tracer_h=p_mflx_tracer_h,
            p_dtime=p_dtime,
            dbl_eps=dbl_eps,
            r_m=r_m,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
