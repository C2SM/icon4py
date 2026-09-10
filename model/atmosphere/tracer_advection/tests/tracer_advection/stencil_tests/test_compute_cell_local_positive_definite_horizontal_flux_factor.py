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

import icon4py.model.common.utils.data_allocation as data_alloc
from icon4py.model.atmosphere.tracer_advection.stencils.compute_cell_local_positive_definite_horizontal_flux_factor import (
    compute_cell_local_positive_definite_horizontal_flux_factor,
)
from icon4py.model.common import dimension as dims
from icon4py.model.testing import stencil_tests


class TestComputeCellLocalPositiveDefiniteHorizontalFluxFactor(stencil_tests.StencilTest):
    PROGRAM = compute_cell_local_positive_definite_horizontal_flux_factor
    OUTPUTS = ("r_m",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        geofac_div: np.ndarray,
        p_cc: np.ndarray,
        p_rhodz_now: np.ndarray,
        p_mflx_tracer_h: np.ndarray,
        p_vn: np.ndarray,
        p_dtime,
        dbl_eps,
        **kwargs,
    ) -> dict:
        c2e = connectivities[dims.C2EDim]
        flux_out = np.zeros_like(p_cc)
        for ie in range(3):
            flux = p_mflx_tracer_h[c2e[:, ie]]
            vn = p_vn[c2e[:, ie]]
            geofac = geofac_div[:, ie, np.newaxis]
            orientation = np.where(geofac > 0.0, 1.0, -1.0)
            is_upwind = np.where(geofac > 0.0, vn >= 0.0, vn < 0.0)
            z_b = orientation * np.maximum(orientation * flux, 0.0)
            flux_out = flux_out + np.where(is_upwind, geofac * p_dtime * z_b, 0.0)
        r_m = np.minimum(1.0, (p_cc * p_rhodz_now) / (flux_out + dbl_eps))
        return dict(r_m=r_m)

    @pytest.fixture
    def input_data(self, grid) -> dict:
        # orientation +-1 times a length/area factor, as geofac_div is
        geofac_div = data_alloc.random_sign(grid, dims.CellDim, dims.C2EDim, dtype=float)
        geofac_div = gtx.as_field(
            (dims.CellDim, dims.C2EDim),
            geofac_div.asnumpy()
            * data_alloc.random_field(grid, dims.CellDim, dims.C2EDim, low=0.5, high=2.0).asnumpy(),
        )
        return dict(
            geofac_div=geofac_div,
            p_cc=data_alloc.random_field(grid, dims.CellDim, dims.KDim, low=0.0, high=0.5),
            p_rhodz_now=data_alloc.random_field(grid, dims.CellDim, dims.KDim, low=0.5, high=1.5),
            p_mflx_tracer_h=data_alloc.random_field(grid, dims.EdgeDim, dims.KDim),
            p_vn=data_alloc.random_field(grid, dims.EdgeDim, dims.KDim),
            r_m=data_alloc.zero_field(grid, dims.CellDim, dims.KDim),
            p_dtime=np.float64(5.0),
            dbl_eps=np.float64(1e-9),
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
