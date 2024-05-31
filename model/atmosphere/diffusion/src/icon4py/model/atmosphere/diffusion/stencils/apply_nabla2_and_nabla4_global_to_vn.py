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
from gt4py.next.ffront.fbuiltins import Field, astype, int32
from model.common.tests import field_aliases as fa

from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _apply_nabla2_and_nabla4_global_to_vn(
    area_edge: fa.EwpField,
    kh_smag_e: Field[[EdgeDim, KDim], vpfloat],
    z_nabla2_e: fa.EKwpField,
    z_nabla4_e2: Field[[EdgeDim, KDim], vpfloat],
    diff_multfac_vn: Field[[KDim], wpfloat],
    vn: fa.EKwpField,
) -> fa.EKwpField:
    kh_smag_e_wp, z_nabla4_e2_wp = astype((kh_smag_e, z_nabla4_e2), wpfloat)

    vn_wp = vn + area_edge * (
        kh_smag_e_wp * z_nabla2_e - diff_multfac_vn * z_nabla4_e2_wp * area_edge
    )
    return vn_wp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def apply_nabla2_and_nabla4_global_to_vn(
    area_edge: fa.EwpField,
    kh_smag_e: Field[[EdgeDim, KDim], vpfloat],
    z_nabla2_e: fa.EKwpField,
    z_nabla4_e2: Field[[EdgeDim, KDim], vpfloat],
    diff_multfac_vn: Field[[KDim], wpfloat],
    vn: fa.EKwpField,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _apply_nabla2_and_nabla4_global_to_vn(
        area_edge,
        kh_smag_e,
        z_nabla2_e,
        z_nabla4_e2,
        diff_multfac_vn,
        vn,
        out=vn,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
