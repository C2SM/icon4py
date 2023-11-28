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
from gt4py.next.ffront.fbuiltins import Field, int32

from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _apply_nabla2_to_vn_in_lateral_boundary(
    z_nabla2_e: Field[[EdgeDim, KDim], wpfloat],
    area_edge: Field[[EdgeDim], wpfloat],
    vn: Field[[EdgeDim, KDim], wpfloat],
    fac_bdydiff_v: wpfloat,
) -> Field[[EdgeDim, KDim], wpfloat]:
    vn_wp = vn + (z_nabla2_e * area_edge * fac_bdydiff_v)
    return vn_wp


@program(grid_type=GridType.UNSTRUCTURED)
def apply_nabla2_to_vn_in_lateral_boundary(
    z_nabla2_e: Field[[EdgeDim, KDim], wpfloat],
    area_edge: Field[[EdgeDim], wpfloat],
    vn: Field[[EdgeDim, KDim], wpfloat],
    fac_bdydiff_v: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _apply_nabla2_to_vn_in_lateral_boundary(z_nabla2_e, area_edge, vn, fac_bdydiff_v, out=vn, domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        })
