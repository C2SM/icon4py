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

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import (
    Field,
    abs,
    astype,
    broadcast,
    int32,
    where,
)
from model.common.tests import field_aliases as fa

from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_maximum_cfl_and_clip_contravariant_vertical_velocity(
    ddqz_z_half: fa.CKvpField,
    z_w_con_c: fa.CKvpField,
    cfl_w_limit: vpfloat,
    dtime: wpfloat,
) -> tuple[
    Field[[CellDim, KDim], bool],
    fa.CKvpField,
    fa.CKvpField,
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
    ddqz_z_half: fa.CKvpField,
    z_w_con_c: fa.CKvpField,
    cfl_clipping: Field[[CellDim, KDim], bool],
    vcfl: fa.CKvpField,
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
