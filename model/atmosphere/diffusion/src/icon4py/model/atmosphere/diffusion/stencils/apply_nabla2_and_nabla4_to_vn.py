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
from gt4py.next.ffront.fbuiltins import Field, broadcast, maximum

from icon4py.model.common.dimension import EdgeDim, KDim


@field_operator
def _apply_nabla2_and_nabla4_to_vn(
    area_edge: Field[[EdgeDim], float],
    kh_smag_e: Field[[EdgeDim, KDim], float],
    z_nabla2_e: Field[[EdgeDim, KDim], float],
    z_nabla4_e2: Field[[EdgeDim, KDim], float],
    diff_multfac_vn: Field[[KDim], float],
    nudgecoeff_e: Field[[EdgeDim], float],
    vn: Field[[EdgeDim, KDim], float],
    nudgezone_diff: float,
) -> Field[[EdgeDim, KDim], float]:
    area_edge_broadcast = broadcast(area_edge, (EdgeDim, KDim))
    vn = vn + area_edge * (
        maximum(nudgezone_diff * nudgecoeff_e, kh_smag_e) * z_nabla2_e
        - area_edge_broadcast * diff_multfac_vn * z_nabla4_e2
    )
    return vn


@program(grid_type=GridType.UNSTRUCTURED)
def apply_nabla2_and_nabla4_to_vn(
    area_edge: Field[[EdgeDim], float],
    kh_smag_e: Field[[EdgeDim, KDim], float],
    z_nabla2_e: Field[[EdgeDim, KDim], float],
    z_nabla4_e2: Field[[EdgeDim, KDim], float],
    diff_multfac_vn: Field[[KDim], float],
    nudgecoeff_e: Field[[EdgeDim], float],
    vn: Field[[EdgeDim, KDim], float],
    nudgezone_diff: float,
):
    _apply_nabla2_and_nabla4_to_vn(
        area_edge,
        kh_smag_e,
        z_nabla2_e,
        z_nabla4_e2,
        diff_multfac_vn,
        nudgecoeff_e,
        vn,
        nudgezone_diff,
        out=vn,
    )
