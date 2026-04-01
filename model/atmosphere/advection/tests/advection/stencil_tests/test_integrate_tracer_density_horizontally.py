# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from collections.abc import Mapping
from typing import Any, cast

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.advection.stencils.integrate_tracer_density_horizontally import (
    integrate_tracer_density_horizontally,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.testing import stencil_tests


class TestIntegrateTracerDensityHorizontally(stencil_tests.StencilTest):
    PROGRAM = integrate_tracer_density_horizontally
    OUTPUTS = (
        "z_rhofluxdiv_c_out",
        "z_fluxdiv_c_dsl",
        "z_rho_new_dsl",
        "z_tracer_new_dsl",
    )

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        p_mass_flx_e: np.ndarray,
        geofac_div: np.ndarray,
        z_rhofluxdiv_c: np.ndarray,
        z_tracer_mflx: np.ndarray,
        z_rho_now: np.ndarray,
        z_tracer_now: np.ndarray,
        z_dtsub: float,
        nsub: gtx.int32,
        **kwargs: Any,
    ) -> dict:
        connectivities = cast(Mapping[gtx.Dimension, np.ndarray], grid.connectivities_asnumpy)
        c2e = connectivities[dims.C2EDim]
        p_mass_flx_e_c2e = p_mass_flx_e[c2e]
        geofac_div = np.expand_dims(geofac_div, axis=-1)
        z_tracer_mflx_c2e = z_tracer_mflx[c2e]

        z_rhofluxdiv_c_out = (
            np.sum(p_mass_flx_e_c2e * geofac_div, axis=1) if nsub == 1 else z_rhofluxdiv_c
        )
        z_fluxdiv_c_dsl = np.sum(z_tracer_mflx_c2e * geofac_div, axis=1)
        z_rho_new_dsl = z_rho_now - z_dtsub * z_rhofluxdiv_c_out
        z_tracer_new_dsl = (z_tracer_now * z_rho_now - z_dtsub * z_fluxdiv_c_dsl) / z_rho_new_dsl

        return dict(
            z_rhofluxdiv_c_out=z_rhofluxdiv_c_out,
            z_fluxdiv_c_dsl=z_fluxdiv_c_dsl,
            z_rho_new_dsl=z_rho_new_dsl,
            z_tracer_new_dsl=z_tracer_new_dsl,
        )

    @stencil_tests.input_data_fixture
    def input_data(self, grid: base.Grid) -> dict:
        p_mass_flx_e = self.data_alloc.random_field(dims.EdgeDim, dims.KDim)
        geofac_div = self.data_alloc.random_field(dims.CellDim, dims.C2EDim)
        z_rhofluxdiv_c = self.data_alloc.random_field(dims.CellDim, dims.KDim)
        z_tracer_mflx = self.data_alloc.random_field(dims.EdgeDim, dims.KDim)
        z_rho_now = self.data_alloc.random_field(dims.CellDim, dims.KDim)
        z_tracer_now = self.data_alloc.random_field(dims.CellDim, dims.KDim)
        z_rhofluxdiv_c_out = self.data_alloc.zero_field(dims.CellDim, dims.KDim)
        z_fluxdiv_c_dsl = self.data_alloc.zero_field(dims.CellDim, dims.KDim)
        z_rho_new_dsl = self.data_alloc.zero_field(dims.CellDim, dims.KDim)
        z_tracer_new_dsl = self.data_alloc.zero_field(dims.CellDim, dims.KDim)
        z_dtsub = 0.5
        nsub = 1
        return dict(
            p_mass_flx_e=p_mass_flx_e,
            geofac_div=geofac_div,
            z_rhofluxdiv_c=z_rhofluxdiv_c,
            z_tracer_mflx=z_tracer_mflx,
            z_rho_now=z_rho_now,
            z_tracer_now=z_tracer_now,
            z_rhofluxdiv_c_out=z_rhofluxdiv_c_out,
            z_fluxdiv_c_dsl=z_fluxdiv_c_dsl,
            z_rho_new_dsl=z_rho_new_dsl,
            z_tracer_new_dsl=z_tracer_new_dsl,
            z_dtsub=z_dtsub,
            nsub=nsub,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
