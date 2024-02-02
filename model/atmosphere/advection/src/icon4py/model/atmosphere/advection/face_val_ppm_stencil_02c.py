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
from gt4py.next import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field

from icon4py.model.common.dimension import CellDim, KDim, Koff


@field_operator
def _face_val_ppm_stencil_02c(
    p_cc: Field[[CellDim, KDim], float],
) -> Field[[CellDim, KDim], float]:
    p_face = p_cc(Koff[-1])
    return p_face


@program(grid_type=GridType.UNSTRUCTURED)
def face_val_ppm_stencil_02c(
    p_cc: Field[[CellDim, KDim], float],
    p_face: Field[[CellDim, KDim], float],
):
    _face_val_ppm_stencil_02c(
        p_cc,
        out=p_face,
    )
