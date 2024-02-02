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
from gt4py.next.ffront.fbuiltins import Field, astype, int32

from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _update_mass_flux_weighted(
    rho_ic: Field[[CellDim, KDim], wpfloat],
    vwind_expl_wgt: Field[[CellDim], wpfloat],
    vwind_impl_wgt: Field[[CellDim], wpfloat],
    w_now: Field[[CellDim, KDim], wpfloat],
    w_new: Field[[CellDim, KDim], wpfloat],
    w_concorr_c: Field[[CellDim, KDim], vpfloat],
    mass_flx_ic: Field[[CellDim, KDim], wpfloat],
    r_nsubsteps: wpfloat,
) -> Field[[CellDim, KDim], wpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_65."""
    w_concorr_c_wp = astype(w_concorr_c, wpfloat)

    mass_flx_ic_wp = mass_flx_ic + (
        r_nsubsteps * rho_ic * (vwind_expl_wgt * w_now + vwind_impl_wgt * w_new - w_concorr_c_wp)
    )
    return mass_flx_ic_wp


@program(grid_type=GridType.UNSTRUCTURED)
def update_mass_flux_weighted(
    rho_ic: Field[[CellDim, KDim], wpfloat],
    vwind_expl_wgt: Field[[CellDim], wpfloat],
    vwind_impl_wgt: Field[[CellDim], wpfloat],
    w_now: Field[[CellDim, KDim], wpfloat],
    w_new: Field[[CellDim, KDim], wpfloat],
    w_concorr_c: Field[[CellDim, KDim], vpfloat],
    mass_flx_ic: Field[[CellDim, KDim], wpfloat],
    r_nsubsteps: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _update_mass_flux_weighted(
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
