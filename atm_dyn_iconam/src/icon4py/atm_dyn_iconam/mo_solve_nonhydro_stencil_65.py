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

from icon4py.common.dimension import CellDim, KDim


@field_operator
def _mo_solve_nonhydro_stencil_65(
    rho_ic: Field[[CellDim, KDim], float],
    vwind_expl_wgt: Field[[CellDim], float],
    vwind_impl_wgt: Field[[CellDim], float],
    w_now: Field[[CellDim, KDim], float],
    w_new: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    mass_flx_ic: Field[[CellDim, KDim], float],
    r_nsubsteps: float,
) -> Field[[CellDim, KDim], float]:
    mass_flx_ic = mass_flx_ic + (
        r_nsubsteps
        * rho_ic
        * (vwind_expl_wgt * w_now + vwind_impl_wgt * w_new - w_concorr_c)
    )
    return mass_flx_ic


@program(grid_type=GridType.UNSTRUCTURED)
def mo_solve_nonhydro_stencil_65(
    rho_ic: Field[[CellDim, KDim], float],
    vwind_expl_wgt: Field[[CellDim], float],
    vwind_impl_wgt: Field[[CellDim], float],
    w_now: Field[[CellDim, KDim], float],
    w_new: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    mass_flx_ic: Field[[CellDim, KDim], float],
    r_nsubsteps: float,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _mo_solve_nonhydro_stencil_65(
        rho_ic,
        vwind_expl_wgt,
        vwind_impl_wgt,
        w_now,
        w_new,
        w_concorr_c,
        mass_flx_ic,
        r_nsubsteps,
        out=mass_flx_ic,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
