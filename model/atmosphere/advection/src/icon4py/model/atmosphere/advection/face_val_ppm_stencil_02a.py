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
import gt4py.next as gtx
from gt4py.next import GridType
from gt4py.next.ffront.decorator import field_operator, program

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff


# TODO: this will have to be removed once domain allows for imports
CellDim = dims.CellDim
KDim = dims.KDim


@field_operator
def _face_val_ppm_stencil_02a(
    p_cc: fa.CellKField[float],
    p_cellhgt_mc_now: fa.CellKField[float],
) -> fa.CellKField[float]:
    p_face = p_cc * (1.0 - (p_cellhgt_mc_now / p_cellhgt_mc_now(Koff[-1]))) + (
        p_cellhgt_mc_now / (p_cellhgt_mc_now(Koff[-1]) + p_cellhgt_mc_now)
    ) * ((p_cellhgt_mc_now / p_cellhgt_mc_now(Koff[-1])) * p_cc + p_cc(Koff[-1]))

    return p_face


@program(grid_type=GridType.UNSTRUCTURED)
def face_val_ppm_stencil_02a(
    p_cc: fa.CellKField[float],
    p_cellhgt_mc_now: fa.CellKField[float],
    p_face: fa.CellKField[float],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _face_val_ppm_stencil_02a(
        p_cc,
        p_cellhgt_mc_now,
        out=p_face,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
