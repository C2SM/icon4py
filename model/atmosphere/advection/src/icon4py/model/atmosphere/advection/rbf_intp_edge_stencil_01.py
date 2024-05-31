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

from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, neighbor_sum

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import E2C2E, E2C2EDim, EdgeDim


@field_operator
def _rbf_intp_edge_stencil_01(
    p_vn_in: fa.EKfloatField,
    ptr_coeff: Field[[EdgeDim, E2C2EDim], float],
) -> fa.EKfloatField:
    p_vt_out = neighbor_sum(p_vn_in(E2C2E) * ptr_coeff, axis=E2C2EDim)
    return p_vt_out


@program
def rbf_intp_edge_stencil_01(
    p_vn_in: fa.EKfloatField,
    ptr_coeff: Field[[EdgeDim, E2C2EDim], float],
    p_vt_out: fa.EKfloatField,
):
    _rbf_intp_edge_stencil_01(p_vn_in, ptr_coeff, out=p_vt_out)
