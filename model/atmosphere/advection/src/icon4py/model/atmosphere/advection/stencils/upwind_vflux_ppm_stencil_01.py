# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program

from icon4py.model.common import field_type_aliases as fa


@field_operator
def _upwind_vflux_ppm_stencil_01(
    z_face_up: fa.CellKField[float],
    z_face_low: fa.CellKField[float],
    p_cc: fa.CellKField[float],
) -> tuple[fa.CellKField[float], fa.CellKField[float]]:
    z_delta_q = 0.5 * (z_face_up - z_face_low)
    z_a1 = p_cc - 0.5 * (z_face_up + z_face_low)

    return z_delta_q, z_a1


@program(grid_type=GridType.UNSTRUCTURED)
def upwind_vflux_ppm_stencil_01(
    z_face_up: fa.CellKField[float],
    z_face_low: fa.CellKField[float],
    p_cc: fa.CellKField[float],
    z_delta_q: fa.CellKField[float],
    z_a1: fa.CellKField[float],
):
    _upwind_vflux_ppm_stencil_01(z_face_up, z_face_low, p_cc, out=(z_delta_q, z_a1))
