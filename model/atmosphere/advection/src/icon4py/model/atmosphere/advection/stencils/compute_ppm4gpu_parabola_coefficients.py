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
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _compute_ppm4gpu_parabola_coefficients(
    z_face_up: fa.CellKField[wpfloat],
    z_face_low: fa.CellKField[wpfloat],
    p_cc: fa.CellKField[wpfloat],
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    z_delta_q = wpfloat(0.5) * (z_face_up - z_face_low)
    z_a1 = p_cc - wpfloat(0.5) * (z_face_up + z_face_low)

    return z_delta_q, z_a1


@program(grid_type=GridType.UNSTRUCTURED)
def compute_ppm4gpu_parabola_coefficients(
    z_face_up: fa.CellKField[wpfloat],
    z_face_low: fa.CellKField[wpfloat],
    p_cc: fa.CellKField[wpfloat],
    z_delta_q: fa.CellKField[wpfloat],
    z_a1: fa.CellKField[wpfloat],
):
    _compute_ppm4gpu_parabola_coefficients(z_face_up, z_face_low, p_cc, out=(z_delta_q, z_a1))
