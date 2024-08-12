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
from gt4py.next.ffront.fbuiltins import astype, int32

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import E2C, E2V
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


# TODO: this will have to be removed once domain allows for imports
EdgeDim = dims.EdgeDim
KDim = dims.KDim


@field_operator
def _compute_horizontal_advection_term_for_vertical_velocity(
    vn_ie: fa.EdgeKField[vpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    w: fa.CellKField[wpfloat],
    z_vt_ie: fa.EdgeKField[vpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    z_w_v: fa.VertexKField[vpfloat],
) -> fa.EdgeKField[vpfloat]:
    """Formerly know as _mo_velocity_advection_stencil_07."""
    z_vt_ie_wp, vn_ie_wp = astype((z_vt_ie, vn_ie), wpfloat)

    z_v_grad_w_wp = vn_ie_wp * inv_dual_edge_length * (
        w(E2C[0]) - w(E2C[1])
    ) + z_vt_ie_wp * inv_primal_edge_length * tangent_orientation * astype(
        z_w_v(E2V[0]) - z_w_v(E2V[1]), wpfloat
    )
    return astype(z_v_grad_w_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_horizontal_advection_term_for_vertical_velocity(
    vn_ie: fa.EdgeKField[vpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    w: fa.CellKField[wpfloat],
    z_vt_ie: fa.EdgeKField[vpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    z_w_v: fa.VertexKField[vpfloat],
    z_v_grad_w: fa.EdgeKField[vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_horizontal_advection_term_for_vertical_velocity(
        vn_ie,
        inv_dual_edge_length,
        w,
        z_vt_ie,
        inv_primal_edge_length,
        tangent_orientation,
        z_w_v,
        out=z_v_grad_w,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
