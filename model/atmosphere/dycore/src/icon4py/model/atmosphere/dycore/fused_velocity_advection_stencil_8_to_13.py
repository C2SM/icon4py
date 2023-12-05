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
from gt4py.next.common import Field, GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import int32, where

from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_08 import (
    _mo_velocity_advection_stencil_08,
)
from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_09 import (
    _mo_velocity_advection_stencil_09,
)
from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_10 import (
    _mo_velocity_advection_stencil_10,
)
from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_11 import (
    _mo_velocity_advection_stencil_11,
)
from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_12 import (
    _mo_velocity_advection_stencil_12,
)
from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_13 import (
    _mo_velocity_advection_stencil_13,
)
from icon4py.model.common.dimension import CEDim, CellDim, EdgeDim, KDim


@field_operator
def _fused_velocity_advection_stencil_8_to_13(
    z_kin_hor_e: Field[[EdgeDim, KDim], float],
    e_bln_c_s: Field[[CEDim], float],
    z_w_concorr_me: Field[[EdgeDim, KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    w: Field[[CellDim, KDim], float],
    z_w_concorr_mc: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    z_ekinh: Field[[CellDim, KDim], float],
    k: Field[[KDim], int32],
    istep: int32,
    nlev: int32,
    nflatlev: int32,
) -> tuple[
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
]:
    z_ekinh = where(
        k < nlev,
        _mo_velocity_advection_stencil_08(z_kin_hor_e, e_bln_c_s),
        z_ekinh,
    )

    #z_w_concorr_mc = (
    #    where(
    #        nflatlev <= k < nlev,
    #        _mo_velocity_advection_stencil_09(z_w_concorr_me, e_bln_c_s),
    #        z_w_concorr_mc,
    #    )
    #    if istep == 1
    #    else z_w_concorr_mc
    #)

    z_w_concorr_mc = _mo_velocity_advection_stencil_09(z_w_concorr_me, e_bln_c_s)

    w_concorr_c = (
        where(
            nflatlev + 1 <= k < nlev,
            _mo_velocity_advection_stencil_10(z_w_concorr_mc, wgtfac_c),
            w_concorr_c,
        )
        if istep == 1
        else w_concorr_c
    )

    z_w_con_c = where(
        k < nlev,
        _mo_velocity_advection_stencil_11(w),
        _mo_velocity_advection_stencil_12(),
    )

    z_w_con_c = where(
        nflatlev + 1 <= k < nlev,
        _mo_velocity_advection_stencil_13(z_w_con_c, w_concorr_c),
        z_w_con_c,
    )

    return z_ekinh, w_concorr_c, z_w_con_c

@field_operator
def _fused_velocity_advection_stencil_8_to_13_restricted(
    z_kin_hor_e: Field[[EdgeDim, KDim], float],
    e_bln_c_s: Field[[CEDim], float],
    z_w_concorr_me: Field[[EdgeDim, KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    w: Field[[CellDim, KDim], float],
    z_w_concorr_mc: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    z_ekinh: Field[[CellDim, KDim], float],
    k: Field[[KDim], int32],
    istep: int32,
    nlev: int32,
    nflatlev: int32,
) -> Field[[CellDim, KDim], float]:

    return _fused_velocity_advection_stencil_8_to_13(
        z_kin_hor_e,
        e_bln_c_s,
        z_w_concorr_me,
        wgtfac_c,
        w,
        z_w_concorr_mc,
        w_concorr_c,
        z_ekinh,
        k,
        istep,
        nlev,
        nflatlev,
    )[2]

@program(grid_type=GridType.UNSTRUCTURED)
def fused_velocity_advection_stencil_8_to_13(
    z_kin_hor_e: Field[[EdgeDim, KDim], float],
    e_bln_c_s: Field[[CEDim], float],
    z_w_concorr_me: Field[[EdgeDim, KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    w: Field[[CellDim, KDim], float],
    z_w_concorr_mc: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    z_ekinh: Field[[CellDim, KDim], float],
    z_w_con_c: Field[[CellDim, KDim], float],
    k: Field[[KDim], int32],
    istep: int32,
    nlev: int32,
    nflatlev: int32,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _fused_velocity_advection_stencil_8_to_13(
        z_kin_hor_e,
        e_bln_c_s,
        z_w_concorr_me,
        wgtfac_c,
        w,
        z_w_concorr_mc,
        w_concorr_c,
        z_ekinh,
        k,
        istep,
        nlev,
        nflatlev,
        out=(z_ekinh, w_concorr_c, z_w_con_c),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end - 1),
        },
    )
    _fused_velocity_advection_stencil_8_to_13_restricted(
        z_kin_hor_e,
        e_bln_c_s,
        z_w_concorr_me,
        wgtfac_c,
        w,
        z_w_concorr_mc,
        w_concorr_c,
        z_ekinh,
        k,
        istep,
        nlev,
        nflatlev,
        out=z_w_con_c,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_end - 1, vertical_end),
        },
    )
