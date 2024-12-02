# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import numpy as np
import pytest

import icon4py.model.common.test_utils.helpers as helpers
from icon4py.model.atmosphere.advection.stencils.compute_antidiffusive_cell_fluxes_and_min_max import (
    compute_antidiffusive_cell_fluxes_and_min_max,
)
from icon4py.model.common import dimension as dims


class TestComputeAntidiffusiveCellFluxesAndMinMax(helpers.StencilTest):
    PROGRAM = compute_antidiffusive_cell_fluxes_and_min_max
    OUTPUTS = (
        "z_mflx_anti_in",
        "z_mflx_anti_out",
        "z_tracer_new_low",
        "z_tracer_max",
        "z_tracer_min",
    )

    @staticmethod
    def reference(
        grid,
        geofac_div: np.ndarray,
        p_rhodz_now: np.ndarray,
        p_rhodz_new: np.ndarray,
        z_mflx_low: np.ndarray,
        z_anti: np.ndarray,
        p_cc: np.ndarray,
        p_dtime: float,
        **kwargs,
    ) -> dict:
        c2e = grid.connectivities[dims.C2EDim]
        z_anti_c2e = z_anti[c2e]

        geofac_div = helpers.reshape(geofac_div, c2e.shape)
        geofac_div = np.expand_dims(geofac_div, axis=-1)

        zero_array = np.zeros(p_rhodz_now.shape)

        z_mflx_anti_1 = p_dtime * geofac_div[:, 0] / p_rhodz_new * z_anti_c2e[:, 0]
        z_mflx_anti_2 = p_dtime * geofac_div[:, 1] / p_rhodz_new * z_anti_c2e[:, 1]
        z_mflx_anti_3 = p_dtime * geofac_div[:, 2] / p_rhodz_new * z_anti_c2e[:, 2]

        z_mflx_anti_in = -1.0 * (
            np.minimum(zero_array, z_mflx_anti_1)
            + np.minimum(zero_array, z_mflx_anti_2)
            + np.minimum(zero_array, z_mflx_anti_3)
        )

        z_mflx_anti_out = (
            np.maximum(zero_array, z_mflx_anti_1)
            + np.maximum(zero_array, z_mflx_anti_2)
            + np.maximum(zero_array, z_mflx_anti_3)
        )

        z_fluxdiv_c = np.sum(z_mflx_low[c2e] * geofac_div, axis=1)

        z_tracer_new_low = (p_cc * p_rhodz_now - p_dtime * z_fluxdiv_c) / p_rhodz_new
        z_tracer_max = np.maximum(p_cc, z_tracer_new_low)
        z_tracer_min = np.minimum(p_cc, z_tracer_new_low)

        return dict(
            z_mflx_anti_in=z_mflx_anti_in,
            z_mflx_anti_out=z_mflx_anti_out,
            z_tracer_new_low=z_tracer_new_low,
            z_tracer_max=z_tracer_max,
            z_tracer_min=z_tracer_min,
        )

    @pytest.fixture
    def input_data(self, grid) -> dict:
        geofac_div = helpers.random_field(grid, dims.CellDim, dims.C2EDim)
        geofac_div_new = helpers.as_1D_sparse_field(geofac_div, dims.CEDim)
        p_rhodz_now = helpers.random_field(grid, dims.CellDim, dims.KDim)
        p_rhodz_new = helpers.random_field(grid, dims.CellDim, dims.KDim)
        z_mflx_low = helpers.random_field(grid, dims.EdgeDim, dims.KDim)
        z_anti = helpers.random_field(grid, dims.EdgeDim, dims.KDim)
        p_cc = helpers.random_field(grid, dims.CellDim, dims.KDim)
        p_dtime = 5.0

        z_mflx_anti_in = helpers.zero_field(grid, dims.CellDim, dims.KDim)
        z_mflx_anti_out = helpers.zero_field(grid, dims.CellDim, dims.KDim)
        z_tracer_new_low = helpers.zero_field(grid, dims.CellDim, dims.KDim)
        z_tracer_max = helpers.zero_field(grid, dims.CellDim, dims.KDim)
        z_tracer_min = helpers.zero_field(grid, dims.CellDim, dims.KDim)

        return dict(
            geofac_div=geofac_div_new,
            p_rhodz_now=p_rhodz_now,
            p_rhodz_new=p_rhodz_new,
            z_mflx_low=z_mflx_low,
            z_anti=z_anti,
            p_cc=p_cc,
            z_mflx_anti_in=z_mflx_anti_in,
            z_mflx_anti_out=z_mflx_anti_out,
            z_tracer_new_low=z_tracer_new_low,
            z_tracer_max=z_tracer_max,
            z_tracer_min=z_tracer_min,
            p_dtime=p_dtime,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
