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
from gt4py.next.ffront.fbuiltins import Field, neighbor_sum

from icon4py.common.dimension import (
    E2C,
    E2EC,
    E2V,
    CellDim,
    E2CDim,
    E2VDim,
    ECDim,
    EdgeDim,
    KDim,
    Koff,
    VertexDim,
    V2CDim,
)


@field_operator
def _mo_icon_interpolation_fields_initalization_stencil(
    edge_cell_length: Field[[EdgeDim], float],
    dual_edge_length: Field[[EdgeDim], float],
) -> tuple[Field[[EdgeDim], float], Field[[EdgeDim], float]]:
    c_lin_e1 = (
        edge_cell_length / dual_edge_length
    )
    c_lin_e2 = 1.0 - c_lin_e1
    return c_lin_e1, c_lin_e2


@program(grid_type=GridType.UNSTRUCTURED)
def mo_icon_interpolation_fields_initalization_stencil(
    edge_cell_length: Field[[EdgeDim], float],
    dual_edge_length: Field[[EdgeDim], float],
    c_lin_e: tuple[Field[[EdgeDim], float], Field[[EdgeDim], float]],
):
    _mo_icon_interpolation_fields_initalization_stencil(
        edge_cell_length,
        dual_edge_length,
        out=c_lin_e,
    )
