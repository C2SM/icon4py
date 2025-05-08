# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.ffront.decorator import field_operator

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import vpfloat


@field_operator
def _compute_first_vertical_derivative(
    cell_k_field: fa.CellKField[vpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
) -> fa.CellKField[vpfloat]:
    first_vertical_derivative_of_cell = (cell_k_field - cell_k_field(Koff[1])) * inv_ddqz_z_full
    return first_vertical_derivative_of_cell
