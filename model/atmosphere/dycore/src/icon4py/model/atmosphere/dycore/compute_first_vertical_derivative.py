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
from gt4py.next.ffront.fbuiltins import int32
from model.common.tests import field_type_aliases as fa

from icon4py.model.common.dimension import CellDim, KDim, Koff
from icon4py.model.common.settings import backend


@field_operator
def _compute_first_vertical_derivative(
    z_exner_ic: fa.CKvpField,
    inv_ddqz_z_full: fa.CKvpField,
) -> fa.CKvpField:
    """Formerly known as _mo_solve_nonhydro_stencil_06."""
    z_dexner_dz_c_1 = (z_exner_ic - z_exner_ic(Koff[1])) * inv_ddqz_z_full
    return z_dexner_dz_c_1


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_first_vertical_derivative(
    z_exner_ic: fa.CKvpField,
    inv_ddqz_z_full: fa.CKvpField,
    z_dexner_dz_c_1: fa.CKvpField,
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
