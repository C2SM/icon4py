# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, neighbor_sum

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import E2C2E, E2C2EDim, EdgeDim


@field_operator
def _rbf_intp_edge_stencil_01(
    p_vn_in: fa.EdgeKField[float],
    ptr_coeff: Field[[EdgeDim, E2C2EDim], float],
) -> fa.EdgeKField[float]:
    p_vt_out = neighbor_sum(p_vn_in(E2C2E) * ptr_coeff, axis=E2C2EDim)
    return p_vt_out


@program
def rbf_intp_edge_stencil_01(
    p_vn_in: fa.EdgeKField[float],
    ptr_coeff: Field[[EdgeDim, E2C2EDim], float],
    p_vt_out: fa.EdgeKField[float],
):
    _rbf_intp_edge_stencil_01(p_vn_in, ptr_coeff, out=p_vt_out)
