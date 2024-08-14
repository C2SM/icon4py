# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, int32, neighbor_sum

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import C2E, C2EDim, CellDim


@field_operator
def _upwind_hflux_miura_cycl_stencil_02(
    nsub: int32,
    p_mass_flx_e: fa.EdgeKField[float],
    geofac_div: Field[[CellDim, C2EDim], float],
    z_rhofluxdiv_c: fa.CellKField[float],
    z_tracer_mflx: fa.EdgeKField[float],
    z_rho_now: fa.CellKField[float],
    z_tracer_now: fa.CellKField[float],
    z_dtsub: float,
) -> tuple[
    fa.CellKField[float],
    fa.CellKField[float],
    fa.CellKField[float],
    fa.CellKField[float],
]:
    z_rhofluxdiv_c_out = (
        neighbor_sum(p_mass_flx_e(C2E) * geofac_div, axis=C2EDim) if nsub == 1 else z_rhofluxdiv_c
    )

    z_fluxdiv_c_dsl = neighbor_sum(z_tracer_mflx(C2E) * geofac_div, axis=C2EDim)

    z_rho_new_dsl = z_rho_now - z_dtsub * z_rhofluxdiv_c_out

    z_tracer_new_dsl = (z_tracer_now * z_rho_now - z_dtsub * z_fluxdiv_c_dsl) / z_rho_new_dsl

    return (z_rhofluxdiv_c_out, z_fluxdiv_c_dsl, z_rho_new_dsl, z_tracer_new_dsl)


@program(grid_type=GridType.UNSTRUCTURED)
def upwind_hflux_miura_cycl_stencil_02(
    nsub: int32,
    p_mass_flx_e: fa.EdgeKField[float],
    geofac_div: Field[[CellDim, C2EDim], float],
    z_rhofluxdiv_c: fa.CellKField[float],
    z_tracer_mflx: fa.EdgeKField[float],
    z_rho_now: fa.CellKField[float],
    z_tracer_now: fa.CellKField[float],
    z_dtsub: float,
    z_rhofluxdiv_c_out: fa.CellKField[float],
    z_fluxdiv_c_dsl: fa.CellKField[float],
    z_rho_new_dsl: fa.CellKField[float],
    z_tracer_new_dsl: fa.CellKField[float],
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
