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

from icon4py.common.dimension import CellDim, KDim


@field_operator
def _apply_nabla2_to_w_in_upper_damping_layer(
    w: Field[[CellDim, KDim], float],
    diff_multfac_n2w: Field[[KDim], float],
    cell_area: Field[[CellDim], float],
    z_nabla2_c: Field[[CellDim, KDim], float],
) -> Field[[CellDim, KDim], float]:
    w = w + diff_multfac_n2w * (cell_area * z_nabla2_c)
    return w


@program
def apply_nabla2_to_w_in_upper_damping_layer(
    w: Field[[CellDim, KDim], float],
    diff_multfac_n2w: Field[[KDim], float],
    cell_area: Field[[CellDim], float],
    z_nabla2_c: Field[[CellDim, KDim], float],
    horizontal_start: int,
    horizontal_end: int,
    vertical_start: int,
    vertical_end: int,
):
    _apply_nabla2_to_w_in_upper_damping_layer(
        w,
        diff_multfac_n2w,
        cell_area,
        z_nabla2_c,
        out=w,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
