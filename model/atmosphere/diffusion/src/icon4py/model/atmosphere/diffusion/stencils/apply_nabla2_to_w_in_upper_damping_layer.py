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
from gt4py.next.ffront.fbuiltins import astype, broadcast, int32
from model.common.tests import field_aliases as fa

from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _apply_nabla2_to_w_in_upper_damping_layer(
    w: fa.CKwpField,
    diff_multfac_n2w: fa.KwpField,
    cell_area: fa.CwpField,
    z_nabla2_c: fa.CKvpField,
) -> fa.CKwpField:
    z_nabla2_c_wp = astype(z_nabla2_c, wpfloat)
    cell_area_tmp = broadcast(cell_area, (CellDim, KDim))

    w_wp = w + diff_multfac_n2w * cell_area_tmp * z_nabla2_c_wp
    return w_wp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def apply_nabla2_to_w_in_upper_damping_layer(
    w: fa.CKwpField,
    diff_multfac_n2w: fa.KwpField,
    cell_area: fa.CwpField,
    z_nabla2_c: fa.CKvpField,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _apply_nabla2_to_w_in_upper_damping_layer(
        w,
        diff_multfac_n2w,
        cell_area,
        z_nabla2_c,
        out=w,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
