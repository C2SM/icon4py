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
def _mo_solve_nonhydro_stencil_50(
    z_rho_expl: Field[[CellDim, KDim], wpfloat],
    z_exner_expl: Field[[CellDim, KDim], wpfloat],
    rho_incr: Field[[CellDim, KDim], vpfloat],
    exner_incr: Field[[CellDim, KDim], vpfloat],
    iau_wgt_dyn: wpfloat,
) -> tuple[Field[[CellDim, KDim], wpfloat], Field[[CellDim, KDim], wpfloat]]:
    rho_incr_wp, exner_incr_wp = astype((rho_incr, exner_incr), wpfloat)

    z_rho_expl_wp = z_rho_expl + iau_wgt_dyn * rho_incr_wp
    z_exner_expl_wp = z_exner_expl + iau_wgt_dyn * exner_incr_wp
    return z_rho_expl_wp, z_exner_expl_wp


@program(grid_type=GridType.UNSTRUCTURED)
def mo_solve_nonhydro_stencil_50(
    z_rho_expl: Field[[CellDim, KDim], wpfloat],
    z_exner_expl: Field[[CellDim, KDim], wpfloat],
    rho_incr: Field[[CellDim, KDim], vpfloat],
    exner_incr: Field[[CellDim, KDim], vpfloat],
    iau_wgt_dyn: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _mo_solve_nonhydro_stencil_50(
        z_rho_expl,
        z_exner_expl,
        rho_incr,
        exner_incr,
        iau_wgt_dyn,
        out=(z_rho_expl, z_exner_expl),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
