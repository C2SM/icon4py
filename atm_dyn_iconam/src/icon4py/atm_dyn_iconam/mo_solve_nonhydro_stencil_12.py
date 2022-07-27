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

from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import Field, FieldOffset

from icon4py.common.dimension import CellDim, KDim

Koff = FieldOffset("Koff", source=KDim, target=(KDim,))


@field_operator
def _mo_solve_nonhydro_stencil_12(
    z_theta_v_pr_ic: Field[[CellDim, KDim], float],
    d2dexdz2_fac1_mc: Field[[CellDim, KDim], float],
    z_rth_pr_2: Field[[CellDim, KDim], float],
    d2dexdz2_fac2_mc: Field[[CellDim, KDim], float],
) -> Field[[CellDim, KDim], float]:
    z_dexner_dz_c_2 = -0.5 * (
        (z_theta_v_pr_ic - z_theta_v_pr_ic(Koff[1])) * d2dexdz2_fac1_mc
        + z_rth_pr_2 * d2dexdz2_fac2_mc
    )
    return z_dexner_dz_c_2


@program
def mo_solve_nonhydro_stencil_12(
    z_theta_v_pr_ic: Field[[CellDim, KDim], float],
    d2dexdz2_fac1_mc: Field[[CellDim, KDim], float],
    z_rth_pr_2: Field[[CellDim, KDim], float],
    d2dexdz2_fac2_mc: Field[[CellDim, KDim], float],
    z_dexner_dz_c_2: Field[[CellDim, KDim], float],
):
    _mo_solve_nonhydro_stencil_12(
        z_theta_v_pr_ic,
        d2dexdz2_fac1_mc,
        z_rth_pr_2,
        d2dexdz2_fac2_mc,
        out=z_dexner_dz_c_2,
    )
