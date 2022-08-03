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
from functional.ffront.fbuiltins import Field

from icon4py.common.dimension import CellDim, KDim, Koff


@field_operator
def _mo_solve_nonhydro_stencil_63(
    inv_ddqz_z_full: Field[[CellDim, KDim], float],
    w: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
) -> Field[[CellDim, KDim], float]:
    z_dwdz_dd = inv_ddqz_z_full * (
        (w - w(Koff[1])) - (w_concorr_c - w_concorr_c(Koff[1]))
    )
    return z_dwdz_dd


@program
def mo_solve_nonhydro_stencil_63(
    inv_ddqz_z_full: Field[[CellDim, KDim], float],
    w: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
    z_dwdz_dd: Field[[CellDim, KDim], float],
):
    _mo_solve_nonhydro_stencil_63(inv_ddqz_z_full, w, w_concorr_c, out=z_dwdz_dd)
