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

from icon4py.model.common.dimension import Koff, CellDim, EdgeDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_dgraddiv_dz_for_full3d_divergence_damping(
    inv_ddqz_z_full: Field[[CellDim, KDim], vpfloat],
    z_dgraddiv_vertical: Field[[CellDim, KDim], vpfloat],
) -> Field[[CellDim, KDim], vpfloat]:
    z_dgraddiv_dz = inv_ddqz_z_full * (z_dgraddiv_vertical - z_dgraddiv_vertical(Koff[1]))
    return z_dgraddiv_dz


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_dgraddiv_dz_for_full3d_divergence_damping(
    inv_ddqz_z_full: Field[[CellDim, KDim], vpfloat],
    z_dgraddiv_vertical: Field[[CellDim, KDim], vpfloat],
    z_dgraddiv_dz: Field[[CellDim, KDim], vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_dgraddiv_dz_for_full3d_divergence_damping(
        inv_ddqz_z_full,
        z_dgraddiv_vertical,
        out=z_dgraddiv_dz,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
