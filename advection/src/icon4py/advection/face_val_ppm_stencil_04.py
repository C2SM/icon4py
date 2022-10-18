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

from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import Field, FieldOffset, broadcast

from icon4py.common.dimension import CellDim, KDim


Koff = FieldOffset("Koff", source=KDim, target=(KDim,))


@field_operator
def _face_val_ppm_stencil_04_p_face_1(
    p_cc: Field[[CellDim, KDim], float],
    p_cellhgt_mc_now: Field[[CellDim, KDim], float],
) -> Field[[CellDim, KDim], float]:

    p_face = p_cc * (
        broadcast(1.0, (CellDim, KDim))
        - (p_cellhgt_mc_now / p_cellhgt_mc_now(Koff[-1]))
    ) + (p_cellhgt_mc_now / (p_cellhgt_mc_now(Koff[-1]) + p_cellhgt_mc_now)) * (
        (p_cellhgt_mc_now / p_cellhgt_mc_now(Koff[-1])) * p_cc + p_cc(Koff[-1])
    )
    return p_face


@field_operator
def _face_val_ppm_stencil_04_p_face_2(
    p_cc: Field[[CellDim, KDim], float],
) -> Field[[CellDim, KDim], float]:
    p_face = p_cc
    return p_face


@program(grid_type="unstructured")
def face_val_ppm_stencil_04(
    p_cc: Field[[CellDim, KDim], float],
    p_cellhgt_mc_now: Field[[CellDim, KDim], float],
    p_face: Field[[CellDim, KDim], float],
    horizontalStart: int,
    horizontalEnd: int,
    verticalStart: int,
    verticalEnd: int,
):
    _face_val_ppm_stencil_04_p_face_1(
        p_cc,
        p_cellhgt_mc_now,
        out=p_face,
        domain={
            CellDim: (horizontalStart, horizontalEnd),
            KDim: (1, 2),
        },
    )

    _face_val_ppm_stencil_04_p_face_1(
        p_cc,
        p_cellhgt_mc_now,
        out=p_face,
        domain={
            CellDim: (horizontalStart, horizontalEnd),
            KDim: (verticalEnd - 1, verticalEnd),
        },
    )

    _face_val_ppm_stencil_04_p_face_2(
        p_cc,
        out=p_face,
        domain={
            CellDim: (horizontalStart, horizontalEnd),
            KDim: (0, 1),
        },
    )
