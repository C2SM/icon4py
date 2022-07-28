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
def _mo_solve_nonhydro_stencil_06(
    z_exner_ic: Field[[CellDim, KDim], float],
    inv_ddqz_z_full: Field[[CellDim, KDim], float],
) -> Field[[CellDim, KDim], float]:
    z_dexner_dz_c_1 = (z_exner_ic - z_exner_ic(Koff[1])) * inv_ddqz_z_full
    return z_dexner_dz_c_1


@program
def mo_solve_nonhydro_stencil_06(
    z_exner_ic: Field[[CellDim, KDim], float],
    inv_ddqz_z_full: Field[[CellDim, KDim], float],
    z_dexner_dz_c_1: Field[[CellDim, KDim], float],
):
    _mo_solve_nonhydro_stencil_06(
        z_exner_ic, inv_ddqz_z_full, out=z_dexner_dz_c_1[:, :-1]
    )
