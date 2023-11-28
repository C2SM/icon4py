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
from gt4py.next.ffront.fbuiltins import Field, astype, neighbor_sum, int32

from icon4py.model.common.dimension import C2CE, C2E, C2EDim, CEDim, CellDim, EdgeDim, KDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _calculate_nabla2_of_theta(
    z_nabla2_e: Field[[EdgeDim, KDim], wpfloat],
    geofac_div: Field[[CEDim], wpfloat],
) -> Field[[CellDim, KDim], vpfloat]:
    z_temp_wp = neighbor_sum(z_nabla2_e(C2E) * geofac_div(C2CE), axis=C2EDim)
    return astype(z_temp_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED)
def calculate_nabla2_of_theta(
    z_nabla2_e: Field[[EdgeDim, KDim], wpfloat],
    geofac_div: Field[[CEDim], wpfloat],
    z_temp: Field[[CellDim, KDim], vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _calculate_nabla2_of_theta(z_nabla2_e, geofac_div, out=z_temp, domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        })
