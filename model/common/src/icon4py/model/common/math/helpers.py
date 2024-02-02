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

from gt4py.next import Field, field_operator

from icon4py.model.common.dimension import CellDim, KDim, Koff
from icon4py.model.common.type_alias import wpfloat


@field_operator
def interpolate_height_levels_for_cell_k(
    half_level_field: Field[[CellDim, KDim], wpfloat]
) -> Field[[CellDim, KDim], wpfloat]:
    """
    Calculate the mean value of adjacent interface levels.

    Computes the average of two adjacent interface levels over a cell field for storage
    in the corresponding full levels.
    Args:
        half_level_field: Field[[CellDim, KDim], wpfloat]

    Returns: Field[[CellDim, KDim], wpfloat] full level field

    """
    return 0.5 * (half_level_field + half_level_field(Koff[+1]))
