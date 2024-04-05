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
from gt4py.next import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, int32

from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.model_backend import backend
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _compute_airmass(
    rho_in: Field[[CellDim, KDim], wpfloat],
    ddqz_z_full_in: Field[[CellDim, KDim], wpfloat],
    deepatmo_t1mc_in: Field[[KDim], wpfloat],
) -> Field[[CellDim, KDim], wpfloat]:
    return rho_in * ddqz_z_full_in * deepatmo_t1mc_in


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_airmass(
    rho_in: Field[[CellDim, KDim], wpfloat],
    ddqz_z_full_in: Field[[CellDim, KDim], wpfloat],
    deepatmo_t1mc_in: Field[[KDim], wpfloat],
    airmass_out: Field[[CellDim, KDim], wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_airmass(
        rho_in,
        ddqz_z_full_in,
        deepatmo_t1mc_in,
        out=airmass_out,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
