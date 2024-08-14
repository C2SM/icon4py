# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.advection.upwind_hflux_miura_cycl_stencil_02 import (
    upwind_hflux_miura_cycl_stencil_02,
)
from icon4py.model.common.dimension import C2EDim, CellDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field


class TestUpwindHfluxMiuraCyclStencil02(StencilTest):
    PROGRAM = upwind_hflux_miura_cycl_stencil_02
    OUTPUTS = (
        "z_rhofluxdiv_c_out",
        "z_fluxdiv_c_dsl",
        "z_rho_new_dsl",
        "z_tracer_new_dsl",
    )

    @staticmethod
    def reference(
        grid,
        nsub: int32,
        p_mass_flx_e: np.array,
        geofac_div: np.array,
        z_rhofluxdiv_c: np.array,
        z_tracer_mflx: np.array,
        z_rho_now: np.array,
        z_tracer_now: np.array,
        z_dtsub: float,
        **kwargs,
    ):
        c2e = grid.connectivities[C2EDim]
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

    @pytest.fixture
    def input_data(self, grid):
        nsub = 1
        p_mass_flx_e = random_field(grid, EdgeDim, KDim)
        geofac_div = random_field(grid, CellDim, C2EDim)
        z_rhofluxdiv_c = random_field(grid, CellDim, KDim)
        z_tracer_mflx = random_field(grid, EdgeDim, KDim)
        z_rho_now = random_field(grid, CellDim, KDim)
        z_tracer_now = random_field(grid, CellDim, KDim)
        z_dtsub = 0.5
        z_rhofluxdiv_c_out = zero_field(grid, CellDim, KDim)
        z_fluxdiv_c_dsl = zero_field(grid, CellDim, KDim)
        z_rho_new_dsl = zero_field(grid, CellDim, KDim)
        z_tracer_new_dsl = zero_field(grid, CellDim, KDim)
        return dict(
            nsub=nsub,
            p_mass_flx_e=p_mass_flx_e,
            geofac_div=geofac_div,
            z_rhofluxdiv_c=z_rhofluxdiv_c,
            z_tracer_mflx=z_tracer_mflx,
            z_rho_now=z_rho_now,
            z_tracer_now=z_tracer_now,
            z_dtsub=z_dtsub,
            z_rhofluxdiv_c_out=z_rhofluxdiv_c_out,
            z_fluxdiv_c_dsl=z_fluxdiv_c_dsl,
            z_rho_new_dsl=z_rho_new_dsl,
            z_tracer_new_dsl=z_tracer_new_dsl,
        )
