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

from icon4py.model.common.dimension import E2C, EdgeDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _calculate_nabla2_for_z(
    kh_smag_e: Field[[EdgeDim, KDim], vpfloat],
    inv_dual_edge_length: fa.EwpField,
    theta_v: fa.CKwpField,
) -> fa.EKwpField:
    kh_smag_e_wp = astype(kh_smag_e, wpfloat)

    z_nabla2_e_wp = kh_smag_e_wp * inv_dual_edge_length * (theta_v(E2C[1]) - theta_v(E2C[0]))
    return z_nabla2_e_wp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def calculate_nabla2_for_z(
    kh_smag_e: Field[[EdgeDim, KDim], vpfloat],
    inv_dual_edge_length: fa.EwpField,
    theta_v: fa.CKwpField,
    z_nabla2_e: fa.EKwpField,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _calculate_nabla2_for_z(
        kh_smag_e,
        inv_dual_edge_length,
        theta_v,
        out=z_nabla2_e,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
