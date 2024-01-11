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

from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, int32, neighbor_sum

from icon4py.model.common.dimension import C2E, C2EDim, CellDim, EdgeDim, KDim


@field_operator
def _upwind_hflux_miura_cycl_stencil_02(
    nsub: int32,
    p_mass_flx_e: Field[[EdgeDim, KDim], float],
    geofac_div: Field[[CellDim, C2EDim], float],
    z_rhofluxdiv_c: Field[[CellDim, KDim], float],
    z_tracer_mflx: Field[[EdgeDim, KDim], float],
    z_rho_now: Field[[CellDim, KDim], float],
    z_tracer_now: Field[[CellDim, KDim], float],
    z_dtsub: float,
) -> tuple[
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
]:
    z_rhofluxdiv_c_out = (
        neighbor_sum(p_mass_flx_e(C2E) * geofac_div, axis=C2EDim)
        if nsub == int32(1)
        else z_rhofluxdiv_c
    )

    z_fluxdiv_c_dsl = neighbor_sum(z_tracer_mflx(C2E) * geofac_div, axis=C2EDim)

    z_rho_new_dsl = z_rho_now - z_dtsub * z_rhofluxdiv_c_out

    z_tracer_new_dsl = (z_tracer_now * z_rho_now - z_dtsub * z_fluxdiv_c_dsl) / z_rho_new_dsl

    return (z_rhofluxdiv_c_out, z_fluxdiv_c_dsl, z_rho_new_dsl, z_tracer_new_dsl)


@program
def upwind_hflux_miura_cycl_stencil_02(
    nsub: int32,
    p_mass_flx_e: Field[[EdgeDim, KDim], float],
    geofac_div: Field[[CellDim, C2EDim], float],
    z_rhofluxdiv_c: Field[[CellDim, KDim], float],
    z_tracer_mflx: Field[[EdgeDim, KDim], float],
    z_rho_now: Field[[CellDim, KDim], float],
    z_tracer_now: Field[[CellDim, KDim], float],
    z_dtsub: float,
    z_rhofluxdiv_c_out: Field[[CellDim, KDim], float],
    z_fluxdiv_c_dsl: Field[[CellDim, KDim], float],
    z_rho_new_dsl: Field[[CellDim, KDim], float],
    z_tracer_new_dsl: Field[[CellDim, KDim], float],
):
    _upwind_hflux_miura_cycl_stencil_02(
        nsub,
        p_mass_flx_e,
        geofac_div,
        z_rhofluxdiv_c,
        z_tracer_mflx,
        z_rho_now,
        z_tracer_now,
        z_dtsub,
        out=(z_rhofluxdiv_c_out, z_fluxdiv_c_dsl, z_rho_new_dsl, z_tracer_new_dsl),
    )
