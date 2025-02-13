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
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, astype, int32

from icon4py.model.common.dimension import E2C, CellDim, EdgeDim, KDim, Koff
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_graddiv_vertical(
    ddqz_z_half: Field[[CellDim, KDim], vpfloat],
    divergence: Field[[CellDim, KDim], vpfloat],
) -> Field[[CellDim, KDim], wpfloat]:
    graddiv_vertical_wp = (divergence(Koff[-1]) - divergence) / astype(ddqz_z_half, wpfloat)
    return astype(graddiv_vertical_wp, vpfloat)


@field_operator
def _compute_graddiv_normal(
    inv_dual_edge_length: Field[[EdgeDim], wpfloat],
    divergence: Field[[CellDim, KDim], vpfloat],
) -> Field[[EdgeDim, KDim], vpfloat]:
    graddiv_normal_wp = inv_dual_edge_length * astype(
        divergence(E2C[1]) - divergence(E2C[0]), wpfloat
    )
    return astype(graddiv_normal_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_graddiv(
    inv_dual_edge_length: Field[[EdgeDim], wpfloat],
    ddqz_z_half: Field[[CellDim, KDim], vpfloat],
    divergence: Field[[CellDim, KDim], vpfloat],
    graddiv_normal: Field[[EdgeDim, KDim], vpfloat],
    graddiv_vertical: Field[[CellDim, KDim], vpfloat],
    edge_horizontal_start: int32,
    edge_horizontal_end: int32,
    cell_horizontal_start: int32,
    cell_horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_graddiv_normal(
        inv_dual_edge_length,
        divergence,
        out=graddiv_normal,
        domain={
            EdgeDim: (edge_horizontal_start, edge_horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
    _compute_graddiv_vertical(
        ddqz_z_half,
        divergence,
        out=graddiv_vertical,
        domain={
            CellDim: (cell_horizontal_start, cell_horizontal_end),
            KDim: (vertical_start + 1, vertical_end),
        },
    )
