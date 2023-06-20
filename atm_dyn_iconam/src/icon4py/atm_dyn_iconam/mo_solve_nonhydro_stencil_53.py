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
from gt4py.next.ffront.decorator import program, scan_operator
from gt4py.next.ffront.fbuiltins import Field

from icon4py.common.dimension import CellDim, KDim


@scan_operator(axis=KDim, forward=False, init=0.0)
def _mo_solve_nonhydro_stencil_53_scan(w_state: float, z_q: float, w: float) -> float:
    return w + w_state * z_q


@program(grid_type=GridType.UNSTRUCTURED)
def mo_solve_nonhydro_stencil_53(
    z_q: Field[[CellDim, KDim], float],
    w: Field[[CellDim, KDim], float],
):
    _mo_solve_nonhydro_stencil_53_scan(z_q, w, out=w[:, 1:])
