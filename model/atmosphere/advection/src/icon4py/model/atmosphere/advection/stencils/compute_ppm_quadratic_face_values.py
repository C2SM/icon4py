# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _compute_ppm_quadratic_face_values(
    p_cc: fa.CellKField[wpfloat],
    p_cellhgt_mc_now: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    p_face = p_cc * (wpfloat(1.0) - (p_cellhgt_mc_now / p_cellhgt_mc_now(Koff[-1]))) + (
        p_cellhgt_mc_now / (p_cellhgt_mc_now(Koff[-1]) + p_cellhgt_mc_now)
    ) * ((p_cellhgt_mc_now / p_cellhgt_mc_now(Koff[-1])) * p_cc + p_cc(Koff[-1]))

    return p_face


@program(grid_type=GridType.UNSTRUCTURED)
def compute_ppm_quadratic_face_values(
    p_cc: fa.CellKField[wpfloat],
    p_cellhgt_mc_now: fa.CellKField[wpfloat],
    p_face: fa.CellKField[wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_ppm_quadratic_face_values(
        p_cc,
        p_cellhgt_mc_now,
        out=p_face,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
