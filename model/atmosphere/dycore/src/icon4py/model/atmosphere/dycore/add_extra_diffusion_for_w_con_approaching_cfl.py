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
    int32,
    minimum,
    neighbor_sum,
    where,
)
from model.common.tests import field_type_aliases as fa

from icon4py.model.common.dimension import C2E2CO, C2E2CODim, CellDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _add_extra_diffusion_for_w_con_approaching_cfl(
    levmask: Field[[KDim], bool],
    cfl_clipping: Field[[CellDim, KDim], bool],
    owner_mask: fa.CboolField,
    z_w_con_c: fa.CKvpField,
    ddqz_z_half: fa.CKvpField,
    area: fa.CwpField,
    geofac_n2s: Field[[CellDim, C2E2CODim], wpfloat],
    w: fa.CKwpField,
    ddt_w_adv: fa.CKvpField,
    scalfac_exdiff: wpfloat,
    cfl_w_limit: vpfloat,
    dtime: wpfloat,
) -> fa.CKvpField:
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


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def add_extra_diffusion_for_w_con_approaching_cfl(
    levmask: Field[[KDim], bool],
    cfl_clipping: Field[[CellDim, KDim], bool],
    owner_mask: fa.CboolField,
    z_w_con_c: fa.CKvpField,
    ddqz_z_half: fa.CKvpField,
    area: fa.CwpField,
    geofac_n2s: Field[[CellDim, C2E2CODim], wpfloat],
    w: fa.CKwpField,
    ddt_w_adv: fa.CKvpField,
    scalfac_exdiff: wpfloat,
    cfl_w_limit: vpfloat,
    dtime: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
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
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
