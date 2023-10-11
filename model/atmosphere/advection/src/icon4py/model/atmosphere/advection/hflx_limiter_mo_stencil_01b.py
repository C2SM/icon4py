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
from gt4py.next.ffront.fbuiltins import Field, broadcast, maximum, minimum, neighbor_sum

from icon4py.model.common.dimension import C2CE, C2E, C2EDim, CEDim, CellDim, EdgeDim, KDim


@field_operator
def _hflx_limiter_mo_stencil_01b(
    geofac_div: Field[[CEDim], float],
    p_rhodz_now: Field[[CellDim, KDim], float],
    p_rhodz_new: Field[[CellDim, KDim], float],
    z_mflx_low: Field[[EdgeDim, KDim], float],
    z_anti: Field[[EdgeDim, KDim], float],
    p_cc: Field[[CellDim, KDim], float],
    p_dtime: float,
) -> tuple[
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
]:
    zero = broadcast(0.0, (CellDim, KDim))

    z_mflx_anti_1 = p_dtime * geofac_div(C2CE[0]) / p_rhodz_new * z_anti(C2E[0])
    z_mflx_anti_2 = p_dtime * geofac_div(C2CE[1]) / p_rhodz_new * z_anti(C2E[1])
    z_mflx_anti_3 = p_dtime * geofac_div(C2CE[2]) / p_rhodz_new * z_anti(C2E[2])

    z_mflx_anti_in = -1.0 * (
        minimum(zero, z_mflx_anti_1) + minimum(zero, z_mflx_anti_2) + minimum(zero, z_mflx_anti_3)
    )

    z_mflx_anti_out = (
        maximum(zero, z_mflx_anti_1) + maximum(zero, z_mflx_anti_2) + maximum(zero, z_mflx_anti_3)
    )

    z_fluxdiv_c = neighbor_sum(z_mflx_low(C2E) * geofac_div(C2CE), axis=C2EDim)

    z_tracer_new_low = (p_cc * p_rhodz_now - p_dtime * z_fluxdiv_c) / p_rhodz_new
    z_tracer_max = maximum(p_cc, z_tracer_new_low)
    z_tracer_min = minimum(p_cc, z_tracer_new_low)

    return (
        z_mflx_anti_in,
        z_mflx_anti_out,
        z_tracer_new_low,
        z_tracer_max,
        z_tracer_min,
    )


@program
def hflx_limiter_mo_stencil_01b(
    geofac_div: Field[[CEDim], float],
    p_rhodz_now: Field[[CellDim, KDim], float],
    p_rhodz_new: Field[[CellDim, KDim], float],
    z_mflx_low: Field[[EdgeDim, KDim], float],
    z_anti: Field[[EdgeDim, KDim], float],
    p_cc: Field[[CellDim, KDim], float],
    p_dtime: float,
    z_mflx_anti_in: Field[[CellDim, KDim], float],
    z_mflx_anti_out: Field[[CellDim, KDim], float],
    z_tracer_new_low: Field[[CellDim, KDim], float],
    z_tracer_max: Field[[CellDim, KDim], float],
    z_tracer_min: Field[[CellDim, KDim], float],
):
    _hflx_limiter_mo_stencil_01b(
        geofac_div,
        p_rhodz_now,
        p_rhodz_new,
        z_mflx_low,
        z_anti,
        p_cc,
        p_dtime,
        out=(
            z_mflx_anti_in,
            z_mflx_anti_out,
            z_tracer_new_low,
            z_tracer_max,
            z_tracer_min,
        ),
    )
