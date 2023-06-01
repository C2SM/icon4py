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
def _mo_solve_nonhydro_stencil_50(
    z_rho_expl: Field[[CellDim, KDim], float],
    z_exner_expl: Field[[CellDim, KDim], float],
    rho_incr: Field[[CellDim, KDim], float],
    exner_incr: Field[[CellDim, KDim], float],
    iau_wgt_dyn: float,
) -> tuple[Field[[CellDim, KDim], float], Field[[CellDim, KDim], float]]:
    z_rho_expl = z_rho_expl + iau_wgt_dyn * rho_incr
    z_exner_expl = z_exner_expl + iau_wgt_dyn * exner_incr
    return z_rho_expl, z_exner_expl


@program
def mo_solve_nonhydro_stencil_50(
    z_rho_expl: Field[[CellDim, KDim], float],
    z_exner_expl: Field[[CellDim, KDim], float],
    rho_incr: Field[[CellDim, KDim], float],
    exner_incr: Field[[CellDim, KDim], float],
    iau_wgt_dyn: float,
):
    _mo_solve_nonhydro_stencil_50(
        z_rho_expl,
        z_exner_expl,
        rho_incr,
        exner_incr,
        iau_wgt_dyn,
        out=(z_rho_expl, z_exner_expl),
    )
