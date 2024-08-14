# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next import GridType
from gt4py.next.ffront.decorator import field_operator, program

from icon4py.model.common import field_type_aliases as fa


@field_operator
def _face_val_ppm_stencil_02b(
    p_cc: fa.CellKField[float],
) -> fa.CellKField[float]:
    p_face = p_cc
    return p_face


@program(grid_type=GridType.UNSTRUCTURED)
def face_val_ppm_stencil_02b(
    p_cc: fa.CellKField[float],
    p_face: fa.CellKField[float],
):
    _face_val_ppm_stencil_02b(
        p_cc,
        out=p_face,
    )
