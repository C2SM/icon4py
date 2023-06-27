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
from gt4py.next.ffront.fbuiltins import Field, neighbor_sum

from icon4py.common.dimension import C2E2CO, C2E2CODim, CellDim, KDim


@field_operator
def _calculate_nabla2_for_w(
    w: Field[[CellDim, KDim], float], geofac_n2s: Field[[CellDim, C2E2CODim], float]
) -> Field[[CellDim, KDim], float]:
    z_nabla2_c = neighbor_sum(w(C2E2CO) * geofac_n2s, axis=C2E2CODim)
    return z_nabla2_c


@program(grid_type=GridType.UNSTRUCTURED)
def calculate_nabla2_for_w(
    w: Field[[CellDim, KDim], float],
    geofac_n2s: Field[[CellDim, C2E2CODim], float],
    z_nabla2_c: Field[[CellDim, KDim], float],
):
    _calculate_nabla2_for_w(w, geofac_n2s, out=z_nabla2_c)
