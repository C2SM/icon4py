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

from icon4py.model.common.dimension import E2C, CellDim, EdgeDim, KDim
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.model_backend import backend


@field_operator
def _compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates(
    inv_dual_edge_length: Field[[EdgeDim], wpfloat],
    z_exner_ex_pr: Field[[CellDim, KDim], vpfloat],
) -> Field[[EdgeDim, KDim], vpfloat]:
    """Formerly konwn as _mo_solve_nonhydro_stencil_18."""
    z_gradh_exner_wp = inv_dual_edge_length * astype(
        z_exner_ex_pr(E2C[1]) - z_exner_ex_pr(E2C[0]), wpfloat
    )
    return astype(z_gradh_exner_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates(
    inv_dual_edge_length: Field[[EdgeDim], wpfloat],
    z_exner_ex_pr: Field[[CellDim, KDim], vpfloat],
    z_gradh_exner: Field[[EdgeDim, KDim], vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates(
        inv_dual_edge_length,
        z_exner_ex_pr,
        out=z_gradh_exner,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
