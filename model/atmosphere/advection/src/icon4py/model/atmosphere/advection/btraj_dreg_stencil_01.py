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
from gt4py.next import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import broadcast, where
from model.common.tests import field_type_aliases as fa

from icon4py.model.common.dimension import EdgeDim, KDim


@field_operator
def _btraj_dreg_stencil_01(
    lcounterclock: bool,
    p_vn: fa.EKfloatField,
    tangent_orientation: fa.EfloatField,
) -> fa.EKboolField:
    tangent_orientation = broadcast(tangent_orientation, (EdgeDim, KDim))
    return where(p_vn * tangent_orientation >= 0.0, lcounterclock, False)


@program(grid_type=GridType.UNSTRUCTURED)
def btraj_dreg_stencil_01(
    lcounterclock: bool,
    p_vn: fa.EKfloatField,
    tangent_orientation: fa.EfloatField,
    lvn_sys_pos: fa.EKboolField,
):
    _btraj_dreg_stencil_01(lcounterclock, p_vn, tangent_orientation, out=lvn_sys_pos)
