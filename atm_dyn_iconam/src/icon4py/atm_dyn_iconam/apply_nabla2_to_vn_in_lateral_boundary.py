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

from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import Field

from icon4py.common.dimension import EdgeDim, KDim


@field_operator
def _apply_nabla2_to_vn_in_lateral_boundary(
    z_nabla2_e: Field[[EdgeDim, KDim], float],
    area_edge: Field[[EdgeDim], float],
    vn: Field[[EdgeDim, KDim], float],
    fac_bdydiff_v: float,
) -> Field[[EdgeDim, KDim], float]:
    vn = vn + (z_nabla2_e * area_edge * fac_bdydiff_v)
    return vn


@program
def apply_nabla2_to_vn_in_lateral_boundary(
    z_nabla2_e: Field[[EdgeDim, KDim], float],
    area_edge: Field[[EdgeDim], float],
    vn: Field[[EdgeDim, KDim], float],
    fac_bdydiff_v: float,
):
    _apply_nabla2_to_vn_in_lateral_boundary(
        z_nabla2_e, area_edge, vn, fac_bdydiff_v, out=vn
    )
