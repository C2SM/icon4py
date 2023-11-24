# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from gt4py.next.ffront.fbuiltins import int32
from gt4py.next.iterator.embedded import StridedNeighborOffsetProvider

from icon4py.model.atmosphere.advection.upwind_hflux_miura_cycl_stencil_02 import (
    upwind_hflux_miura_cycl_stencil_02,
)
from icon4py.model.common.dimension import C2EDim, CEDim, CellDim, EdgeDim, KDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.test_utils.helpers import random_field


def upwind_hflux_miura_cycl_stencil_02_numpy(
    c2e: np.array,
    nsub: int32,
    p_mass_flx_e: np.array,
    geofac_div: np.array,
    z_rhofluxdiv_c: np.array,
    z_tracer_mflx: np.array,
    z_rho_now: np.array,
    z_tracer_now: np.array,
    z_dtsub: float,
):
    p_mass_flx_e_c2e = p_mass_flx_e[c2e]
    geofac_div = np.expand_dims(geofac_div, axis=-1)
    z_tracer_mflx_c2e = z_tracer_mflx[c2e]

    z_rhofluxdiv_c_out = (
        np.sum(p_mass_flx_e_c2e * geofac_div, axis=1) if nsub == int32(1) else z_rhofluxdiv_c
    )
    z_fluxdiv_c_dsl = np.sum(z_tracer_mflx_c2e * geofac_div, axis=1)

    z_rho_new_dsl = z_rho_now - z_dtsub * z_rhofluxdiv_c_out

    z_tracer_new_dsl = (z_tracer_now * z_rho_now - z_dtsub * z_fluxdiv_c_dsl) / z_rho_new_dsl

    return (z_rhofluxdiv_c_out, z_fluxdiv_c_dsl, z_rho_new_dsl, z_tracer_new_dsl)


def test_upwind_hflux_miura_cycl_stencil_02(backend):
    grid = SimpleGrid()
    nsub = int32(1)
    p_mass_flx_e = random_field(grid, EdgeDim, KDim)
    geofac_div = random_field(grid, CellDim, C2EDim)
    z_rhofluxdiv_c = random_field(grid, CellDim, KDim)
    z_tracer_mflx = random_field(grid, EdgeDim, KDim)
    z_rho_now = random_field(grid, CellDim, KDim)
    z_tracer_now = random_field(grid, CellDim, KDim)
    z_dtsub = 0.5
    z_rhofluxdiv_c_out = random_field(grid, CellDim, KDim)
    z_fluxdiv_c_dsl = random_field(grid, CellDim, KDim)
    z_rho_new_dsl = random_field(grid, CellDim, KDim)
    z_tracer_new_dsl = random_field(grid, CellDim, KDim)

    ref_1, ref_2, ref_3, ref_4 = upwind_hflux_miura_cycl_stencil_02_numpy(
        grid.connectivities[C2EDim],
        nsub,
        p_mass_flx_e.asnumpy(),
        geofac_div.asnumpy(),
        z_rhofluxdiv_c.asnumpy(),
        z_tracer_mflx.asnumpy(),
        z_rho_now.asnumpy(),
        z_tracer_now.asnumpy(),
        z_dtsub,
    )

    upwind_hflux_miura_cycl_stencil_02.with_backend(backend)(
        nsub,
        p_mass_flx_e,
        geofac_div,
        z_rhofluxdiv_c,
        z_tracer_mflx,
        z_rho_now,
        z_tracer_now,
        z_dtsub,
        z_rhofluxdiv_c_out,
        z_fluxdiv_c_dsl,
        z_rho_new_dsl,
        z_tracer_new_dsl,
        offset_provider={
            "C2CE": StridedNeighborOffsetProvider(CellDim, CEDim, grid.size[C2EDim]),
            "C2E": grid.get_offset_provider("C2E"),
        },
    )
    assert np.allclose(ref_1, z_rhofluxdiv_c_out.asnumpy())
    assert np.allclose(ref_2, z_fluxdiv_c_dsl.asnumpy())
    assert np.allclose(ref_3, z_rho_new_dsl.asnumpy())
    assert np.allclose(ref_4, z_tracer_new_dsl.asnumpy())
