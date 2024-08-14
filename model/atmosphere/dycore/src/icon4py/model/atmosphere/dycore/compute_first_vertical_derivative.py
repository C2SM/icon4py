# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import CellDim, KDim, Koff
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat


@field_operator
def _compute_first_vertical_derivative(
    z_exner_ic: fa.CellKField[vpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
) -> fa.CellKField[vpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_06."""
    z_dexner_dz_c_1 = (z_exner_ic - z_exner_ic(Koff[1])) * inv_ddqz_z_full
    return z_dexner_dz_c_1


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_first_vertical_derivative(
    z_exner_ic: fa.CellKField[vpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    z_dexner_dz_c_1: fa.CellKField[vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_first_vertical_derivative(
        z_exner_ic,
        inv_ddqz_z_full,
        out=z_dexner_dz_c_1,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
