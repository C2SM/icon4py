# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, broadcast, int32, maximum, minimum, neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2CE, C2E
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_antidiffusive_cell_fluxes_and_min_max(
    geofac_div: Field[[dims.CEDim], wpfloat],
    p_rhodz_now: fa.CellKField[wpfloat],
    p_rhodz_new: fa.CellKField[wpfloat],
    z_mflx_low: fa.EdgeKField[wpfloat],
    z_anti: fa.EdgeKField[wpfloat],
    p_cc: fa.CellKField[wpfloat],
    p_dtime: wpfloat,
) -> tuple[
    fa.CellKField[vpfloat],
    fa.CellKField[vpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[vpfloat],
    fa.CellKField[vpfloat],
]:
    zero = broadcast(vpfloat(0.0), (dims.CellDim, dims.KDim))

    z_mflx_anti_1 = p_dtime * geofac_div(C2CE[0]) / p_rhodz_new * z_anti(C2E[0])
    z_mflx_anti_2 = p_dtime * geofac_div(C2CE[1]) / p_rhodz_new * z_anti(C2E[1])
    z_mflx_anti_3 = p_dtime * geofac_div(C2CE[2]) / p_rhodz_new * z_anti(C2E[2])

    z_mflx_anti_in = -vpfloat(1.0) * (
        minimum(zero, z_mflx_anti_1) + minimum(zero, z_mflx_anti_2) + minimum(zero, z_mflx_anti_3)
    )

    z_mflx_anti_out = (
        maximum(zero, z_mflx_anti_1) + maximum(zero, z_mflx_anti_2) + maximum(zero, z_mflx_anti_3)
    )

    z_fluxdiv_c = neighbor_sum(z_mflx_low(C2E) * geofac_div(C2CE), axis=dims.C2EDim)

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
def compute_antidiffusive_cell_fluxes_and_min_max(
    geofac_div: Field[[dims.CEDim], wpfloat],
    p_rhodz_now: fa.CellKField[wpfloat],
    p_rhodz_new: fa.CellKField[wpfloat],
    z_mflx_low: fa.EdgeKField[wpfloat],
    z_anti: fa.EdgeKField[wpfloat],
    p_cc: fa.CellKField[wpfloat],
    p_dtime: wpfloat,
    z_mflx_anti_in: fa.CellKField[vpfloat],
    z_mflx_anti_out: fa.CellKField[vpfloat],
    z_tracer_new_low: fa.CellKField[wpfloat],
    z_tracer_max: fa.CellKField[vpfloat],
    z_tracer_min: fa.CellKField[vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_antidiffusive_cell_fluxes_and_min_max(
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
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
