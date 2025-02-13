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
from gt4py.next.ffront.fbuiltins import Field, broadcast, int32

from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _apply_nabla2_to_vn_in_lateral_boundary(
    z_nabla2_e: Field[[EdgeDim, KDim], wpfloat],
    area_edge: Field[[EdgeDim], wpfloat],
    vn: Field[[EdgeDim, KDim], wpfloat],
    fac_bdydiff_v: wpfloat,
) -> tuple[
    Field[[EdgeDim, KDim], wpfloat],
    Field[[EdgeDim, KDim], wpfloat],
    Field[[EdgeDim, KDim], wpfloat],
]:
    vn_wp = vn + (area_edge * fac_bdydiff_v * z_nabla2_e)
    nabla2_diff = area_edge * fac_bdydiff_v * z_nabla2_e
    nabla4_diff = broadcast(wpfloat("0.0"), (EdgeDim, KDim))
    return vn_wp, nabla2_diff, nabla4_diff


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def apply_nabla2_to_vn_in_lateral_boundary(
    z_nabla2_e: Field[[EdgeDim, KDim], wpfloat],
    area_edge: Field[[EdgeDim], wpfloat],
    vn: Field[[EdgeDim, KDim], wpfloat],
    nabla2_diff: Field[[EdgeDim, KDim], wpfloat],
    nabla4_diff: Field[[EdgeDim, KDim], wpfloat],
    fac_bdydiff_v: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _apply_nabla2_to_vn_in_lateral_boundary(
        z_nabla2_e,
        area_edge,
        vn,
        fac_bdydiff_v,
        out=(vn, nabla2_diff, nabla4_diff),
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
