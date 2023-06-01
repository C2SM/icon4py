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
from gt4py.next.ffront.fbuiltins import Field
from icon4py.model.common.dimension import CellDim, KDim


@field_operator
def _mo_solve_nonhydro_stencil_58(
    z_contr_w_fl_l: Field[[CellDim, KDim], float],
    rho_ic: Field[[CellDim, KDim], float],
    vwind_impl_wgt: Field[[CellDim], float],
    w: Field[[CellDim, KDim], float],
    mass_flx_ic: Field[[CellDim, KDim], float],
    r_nsubsteps: float,
) -> Field[[CellDim, KDim], float]:
    mass_flx_ic = mass_flx_ic + (
        r_nsubsteps * (z_contr_w_fl_l + rho_ic * vwind_impl_wgt * w)
    )
    return mass_flx_ic


@program
def mo_solve_nonhydro_stencil_58(
    z_contr_w_fl_l: Field[[CellDim, KDim], float],
    rho_ic: Field[[CellDim, KDim], float],
    vwind_impl_wgt: Field[[CellDim], float],
    w: Field[[CellDim, KDim], float],
    mass_flx_ic: Field[[CellDim, KDim], float],
    r_nsubsteps: float,
):
    _mo_solve_nonhydro_stencil_58(
        z_contr_w_fl_l,
        rho_ic,
        vwind_impl_wgt,
        w,
        mass_flx_ic,
        r_nsubsteps,
        out=mass_flx_ic,
    )
