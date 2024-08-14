# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import (
    abs,
    astype,
    broadcast,
    int32,
    where,
)

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_maximum_cfl_and_clip_contravariant_vertical_velocity(
    ddqz_z_half: fa.CellKField[vpfloat],
    z_w_con_c: fa.CellKField[vpfloat],
    cfl_w_limit: vpfloat,
    dtime: wpfloat,
) -> tuple[
    fa.CellKField[bool],
    fa.CellKField[vpfloat],
    fa.CellKField[vpfloat],
]:
    """Formerly know as _mo_velocity_advection_stencil_14."""
    z_w_con_c_wp, ddqz_z_half_wp = astype((z_w_con_c, ddqz_z_half), wpfloat)

    cfl_clipping = where(
        abs(z_w_con_c) > cfl_w_limit * ddqz_z_half,
        broadcast(True, (CellDim, KDim)),
        False,
    )

    vcfl = where(cfl_clipping, z_w_con_c_wp * dtime / ddqz_z_half_wp, wpfloat("0.0"))
    vcfl_vp = astype(vcfl, vpfloat)

    z_w_con_c_wp = where(
        (cfl_clipping) & (vcfl_vp < -vpfloat("0.85")),
        astype(-vpfloat("0.85") * ddqz_z_half, wpfloat) / dtime,
        z_w_con_c_wp,
    )

    z_w_con_c_wp = where(
        (cfl_clipping) & (vcfl_vp > vpfloat("0.85")),
        astype(vpfloat("0.85") * ddqz_z_half, wpfloat) / dtime,
        z_w_con_c_wp,
    )

    return cfl_clipping, vcfl_vp, astype(z_w_con_c_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_maximum_cfl_and_clip_contravariant_vertical_velocity(
    ddqz_z_half: fa.CellKField[vpfloat],
    z_w_con_c: fa.CellKField[vpfloat],
    cfl_clipping: fa.CellKField[bool],
    vcfl: fa.CellKField[vpfloat],
    cfl_w_limit: vpfloat,
    dtime: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_maximum_cfl_and_clip_contravariant_vertical_velocity(
        ddqz_z_half,
        z_w_con_c,
        cfl_w_limit,
        dtime,
        out=(cfl_clipping, vcfl, z_w_con_c),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
