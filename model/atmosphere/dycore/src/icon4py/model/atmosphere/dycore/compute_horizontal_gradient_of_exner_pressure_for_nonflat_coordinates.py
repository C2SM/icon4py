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
from gt4py.next.ffront.fbuiltins import Field, astype, int32, neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import E2C, E2CDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates(
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    z_exner_ex_pr: fa.CellKField[vpfloat],
    ddxn_z_full: fa.EdgeKField[vpfloat],
    c_lin_e: Field[[dims.EdgeDim, E2CDim], wpfloat],
    z_dexner_dz_c_1: fa.CellKField[vpfloat],
) -> fa.EdgeKField[vpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_19."""
    ddxn_z_full_wp, z_dexner_dz_c_1_wp = astype((ddxn_z_full, z_dexner_dz_c_1), wpfloat)

    z_gradh_exner_wp = inv_dual_edge_length * (
        astype(z_exner_ex_pr(E2C[1]) - z_exner_ex_pr(E2C[0]), wpfloat)
    ) - ddxn_z_full_wp * neighbor_sum(z_dexner_dz_c_1_wp(E2C) * c_lin_e, axis=E2CDim)
    return astype(z_gradh_exner_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates(
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    z_exner_ex_pr: fa.CellKField[vpfloat],
    ddxn_z_full: fa.EdgeKField[vpfloat],
    c_lin_e: Field[[dims.EdgeDim, E2CDim], wpfloat],
    z_dexner_dz_c_1: fa.CellKField[vpfloat],
    z_gradh_exner: fa.EdgeKField[vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates(
        inv_dual_edge_length,
        z_exner_ex_pr,
        ddxn_z_full,
        c_lin_e,
        z_dexner_dz_c_1,
        out=z_gradh_exner,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
