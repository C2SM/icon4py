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

from icon4py.model.common.dimension import EdgeDim, KDim


@field_operator
def _mo_solve_nonhydro_stencil_29(
    grf_tend_vn: Field[[EdgeDim, KDim], float],
    vn_now: Field[[EdgeDim, KDim], float],
    dtime: float,
) -> Field[[EdgeDim, KDim], float]:
    vn_new = vn_now + dtime * grf_tend_vn
    return vn_new


@program
def mo_solve_nonhydro_stencil_29(
    grf_tend_vn: Field[[EdgeDim, KDim], float],
    vn_now: Field[[EdgeDim, KDim], float],
    vn_new: Field[[EdgeDim, KDim], float],
    dtime: float,
):
    _mo_solve_nonhydro_stencil_29(grf_tend_vn, vn_now, dtime, out=vn_new)
