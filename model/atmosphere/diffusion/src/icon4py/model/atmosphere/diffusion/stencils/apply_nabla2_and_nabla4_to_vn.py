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
from gt4py.next.ffront.fbuiltins import Field, broadcast, astype, maximum

from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _apply_nabla2_and_nabla4_to_vn(
    area_edge: Field[[EdgeDim], wpfloat],
    kh_smag_e: Field[[EdgeDim, KDim], vpfloat],
    z_nabla2_e: Field[[EdgeDim, KDim], wpfloat],
    z_nabla4_e2: Field[[EdgeDim, KDim], vpfloat],
    diff_multfac_vn: Field[[KDim], wpfloat],
    nudgecoeff_e: Field[[EdgeDim], wpfloat],
    vn: Field[[EdgeDim, KDim], wpfloat],
    nudgezone_diff: vpfloat,
) -> Field[[EdgeDim, KDim], wpfloat]:
    kh_smag_e_wp, z_nabla4_e2_wp, nudgezone_diff_wp = astype(
        (kh_smag_e, z_nabla4_e2, nudgezone_diff), wpfloat
    )
    area_edge_broadcast = broadcast(area_edge, (EdgeDim, KDim))


    vn_wp = vn + area_edge * (
        maximum(nudgezone_diff_wp * nudgecoeff_e, kh_smag_e_wp) * z_nabla2_e
        - area_edge_broadcast * diff_multfac_vn * z_nabla4_e2_wp
    )
    return vn_wp


@program(grid_type=GridType.UNSTRUCTURED)
def apply_nabla2_and_nabla4_to_vn(
    area_edge: Field[[EdgeDim], wpfloat],
    kh_smag_e: Field[[EdgeDim, KDim], vpfloat],
    z_nabla2_e: Field[[EdgeDim, KDim], wpfloat],
    z_nabla4_e2: Field[[EdgeDim, KDim], vpfloat],
    diff_multfac_vn: Field[[KDim], wpfloat],
    nudgecoeff_e: Field[[EdgeDim], wpfloat],
    vn: Field[[EdgeDim, KDim], wpfloat],
    nudgezone_diff: vpfloat,
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
