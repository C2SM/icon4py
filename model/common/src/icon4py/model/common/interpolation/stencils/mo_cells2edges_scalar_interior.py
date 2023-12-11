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
from gt4py.next.ffront.fbuiltins import Field, int32, neighbor_sum

from icon4py.model.common.dimension import CellDim, EdgeDim, KDim, E2CDim, E2C

@field_operator
def _mo_cells2edges_scalar_interior(
    cells2edges_interpolation_coeff: Field[[EdgeDim, E2CDim], float],
    cell_scalar: Field[[CellDim, KDim], float],
) -> Field[[EdgeDim, KDim], float]:
    edge_scalar = neighbor_sum(cells2edges_interpolation_coeff * cell_scalar(E2C), axis=E2CDim)
    return edge_scalar


@program(grid_type=GridType.UNSTRUCTURED)
def mo_cells2edges_scalar_interior(
    cells2edges_interpolation_coeff: Field[[EdgeDim, E2CDim], float],
    cell_scalar: Field[[CellDim, KDim], float],
    edge_scalar: Field[[EdgeDim, KDim], float],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _mo_cells2edges_scalar_interior(
        cells2edges_interpolation_coeff,
        cell_scalar,
        out=edge_scalar,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
