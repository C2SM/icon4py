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

from gt4py.next.common import Field, GridType
from gt4py.next.ffront.decorator import field_operator, program

from icon4py.model.common.dimension import CellDim, KDim


@field_operator
def _upwind_vflux_ppm_stencil_01(
    z_face_up: Field[[CellDim, KDim], float],
    z_face_low: Field[[CellDim, KDim], float],
    p_cc: Field[[CellDim, KDim], float],
) -> tuple[Field[[CellDim, KDim], float], Field[[CellDim, KDim], float]]:
    z_delta_q = 0.5 * (z_face_up - z_face_low)
    z_a1 = p_cc - 0.5 * (z_face_up + z_face_low)

    return z_delta_q, z_a1


@program(grid_type=GridType.UNSTRUCTURED)
def upwind_vflux_ppm_stencil_01(
    z_face_up: Field[[CellDim, KDim], float],
    z_face_low: Field[[CellDim, KDim], float],
    p_cc: Field[[CellDim, KDim], float],
    z_delta_q: Field[[CellDim, KDim], float],
    z_a1: Field[[CellDim, KDim], float],
):
    _upwind_vflux_ppm_stencil_01(z_face_up, z_face_low, p_cc, out=(z_delta_q, z_a1))
