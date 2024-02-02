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
from gt4py.next.ffront.fbuiltins import Field, int32

from icon4py.model.atmosphere.dycore.update_wind import (
    _update_wind,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _update_densety_exener_wind(
    rho_now: Field[[CellDim, KDim], wpfloat],
    grf_tend_rho: Field[[CellDim, KDim], wpfloat],
    theta_v_now: Field[[CellDim, KDim], wpfloat],
    grf_tend_thv: Field[[CellDim, KDim], wpfloat],
    w_now: Field[[CellDim, KDim], wpfloat],
    grf_tend_w: Field[[CellDim, KDim], wpfloat],
    dtime: wpfloat,
) -> tuple[
    Field[[CellDim, KDim], wpfloat],
    Field[[CellDim, KDim], wpfloat],
    Field[[CellDim, KDim], wpfloat],
]:
    '''Formerly known as _mo_solve_nonhydro_stencil_61.'''
    rho_new_wp = rho_now + dtime * grf_tend_rho
    exner_new_wp = theta_v_now + dtime * grf_tend_thv
    w_new_wp = _update_wind(w_now=w_now, grf_tend_w=grf_tend_w, dtime=dtime)
    return rho_new_wp, exner_new_wp, w_new_wp


@program(grid_type=GridType.UNSTRUCTURED)
def update_densety_exener_wind(
    rho_now: Field[[CellDim, KDim], wpfloat],
    grf_tend_rho: Field[[CellDim, KDim], wpfloat],
    theta_v_now: Field[[CellDim, KDim], wpfloat],
    grf_tend_thv: Field[[CellDim, KDim], wpfloat],
    w_now: Field[[CellDim, KDim], wpfloat],
    grf_tend_w: Field[[CellDim, KDim], wpfloat],
    rho_new: Field[[CellDim, KDim], wpfloat],
    exner_new: Field[[CellDim, KDim], wpfloat],
    w_new: Field[[CellDim, KDim], wpfloat],
    dtime: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _update_densety_exener_wind(
        rho_now,
        grf_tend_rho,
        theta_v_now,
        grf_tend_thv,
        w_now,
        grf_tend_w,
        dtime,
        out=(rho_new, exner_new, w_new),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
