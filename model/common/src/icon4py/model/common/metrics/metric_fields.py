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

from gt4py.next import Field, GridType, int32, program

from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.math.helpers import interpolate_height_levels_for_cell_k
from icon4py.model.common.type_alias import wpfloat


@program(grid_type=GridType.UNSTRUCTURED)
def compute_z_mc(
    z_ifc: Field[[CellDim, KDim], wpfloat],
    z_mc: Field[[CellDim, KDim], wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    """Compute the geometric height of full levels from the geometric height of half levels (z_ifc).

    This assumes that the input field z_ifc is defined on half levels (KHalfDim) and the
    returned fields is defined on full levels (KDim)

    Args:
        z_ifc: Field[[CellDim, KDim], wpfloat] geometric height on half levels
        z_mc: Field[[CellDim, KDim], wpfloat] output, geometric height defined on full levels
        horizontal_start:int32 start index of horizontal domain
        horizontal_end:int32 end index of horizontal domain
        vertical_start:int32 start index of vertical domain
        vertical_end:int32 end index of vertical domain

    Returns:
    """
    interpolate_height_levels_for_cell_k(
        z_ifc,
        out=z_mc,
        domain={CellDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)},
    )

    """

    """
