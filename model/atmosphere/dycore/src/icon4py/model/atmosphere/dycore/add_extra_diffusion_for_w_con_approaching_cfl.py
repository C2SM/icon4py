# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import (
    abs,
    astype,
    minimum,
    neighbor_sum,
    where,
)

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2E2CO, C2E2CODim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _add_extra_diffusion_for_w_con_approaching_cfl(
    levmask: gtx.Field[gtx.Dims[dims.KDim], bool],
    cfl_clipping: fa.CellKField[bool],
    owner_mask: fa.CellField[bool],
    z_w_con_c: fa.CellKField[vpfloat],
    ddqz_z_half: fa.CellKField[vpfloat],
    area: fa.CellField[wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, C2E2CODim], wpfloat],
    w: fa.CellKField[wpfloat],
    ddt_w_adv: fa.CellKField[vpfloat],
    scalfac_exdiff: wpfloat,
    cfl_w_limit: vpfloat,
    dtime: wpfloat,
) -> fa.CellKField[vpfloat]:
    """Formerly known as _mo_velocity_advection_stencil_18."""
    z_w_con_c_wp, ddqz_z_half_wp, ddt_w_adv_wp, cfl_w_limit_wp = astype(
        (z_w_con_c, ddqz_z_half, ddt_w_adv, cfl_w_limit), wpfloat
    )

    difcoef = where(
        levmask & cfl_clipping & owner_mask,
        scalfac_exdiff
        * minimum(
            wpfloat("0.85") - cfl_w_limit_wp * dtime,
            abs(z_w_con_c_wp) * dtime / ddqz_z_half_wp - cfl_w_limit_wp * dtime,
        ),
        wpfloat("0.0"),
    )

    ddt_w_adv_wp = where(
        levmask & cfl_clipping & owner_mask,
        ddt_w_adv_wp + difcoef * area * neighbor_sum(w(C2E2CO) * geofac_n2s, axis=C2E2CODim),
        ddt_w_adv_wp,
    )

    return astype(ddt_w_adv_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED)
def add_extra_diffusion_for_w_con_approaching_cfl(
    levmask: gtx.Field[gtx.Dims[dims.KDim], bool],
    cfl_clipping: fa.CellKField[bool],
    owner_mask: fa.CellField[bool],
    z_w_con_c: fa.CellKField[vpfloat],
    ddqz_z_half: fa.CellKField[vpfloat],
    area: fa.CellField[wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, C2E2CODim], wpfloat],
    w: fa.CellKField[wpfloat],
    ddt_w_adv: fa.CellKField[vpfloat],
    scalfac_exdiff: wpfloat,
    cfl_w_limit: vpfloat,
    dtime: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _add_extra_diffusion_for_w_con_approaching_cfl(
        levmask,
        cfl_clipping,
        owner_mask,
        z_w_con_c,
        ddqz_z_half,
        area,
        geofac_n2s,
        w,
        ddt_w_adv,
        scalfac_exdiff,
        cfl_w_limit,
        dtime,
        out=ddt_w_adv,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
