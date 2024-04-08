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

from icon4py.model.atmosphere.dycore.set_cell_kdim_field_to_zero_wp import (
    _set_cell_kdim_field_to_zero_wp,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.model_backend import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _set_lower_boundary_condition_for_w_and_contravariant_correction(
    w_concorr_c: Field[[CellDim, KDim], vpfloat],
) -> tuple[Field[[CellDim, KDim], wpfloat], Field[[CellDim, KDim], wpfloat]]:
    """Formerly known as _mo_solve_nonhydro_stencil_47."""
    w_concorr_c_wp = astype(w_concorr_c, wpfloat)

    w_nnew_wp = w_concorr_c_wp
    z_contr_w_fl_l_wp = _set_cell_kdim_field_to_zero_wp()
    return w_nnew_wp, z_contr_w_fl_l_wp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def set_lower_boundary_condition_for_w_and_contravariant_correction(
    w_nnew: Field[[CellDim, KDim], wpfloat],
    z_contr_w_fl_l: Field[[CellDim, KDim], wpfloat],
    w_concorr_c: Field[[CellDim, KDim], vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _set_lower_boundary_condition_for_w_and_contravariant_correction(
        w_concorr_c,
        out=(w_nnew, z_contr_w_fl_l),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
