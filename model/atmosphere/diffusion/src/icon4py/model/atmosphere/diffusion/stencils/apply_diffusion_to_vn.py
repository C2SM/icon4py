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
from gt4py.next.ffront.fbuiltins import Field, int32, where

from icon4py.model.atmosphere.diffusion.stencils.apply_nabla2_and_nabla4_global_to_vn import (
    _apply_nabla2_and_nabla4_global_to_vn,
)
from icon4py.model.atmosphere.diffusion.stencils.apply_nabla2_and_nabla4_to_vn import (
    _apply_nabla2_and_nabla4_to_vn,
)
from icon4py.model.atmosphere.diffusion.stencils.apply_nabla2_to_vn_in_lateral_boundary import (
    _apply_nabla2_to_vn_in_lateral_boundary,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla4 import _calculate_nabla4
from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import ECVDim, EdgeDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _apply_diffusion_to_vn(
    u_vert: fa.VKvpField,
    v_vert: fa.VKvpField,
    primal_normal_vert_v1: Field[[ECVDim], wpfloat],
    primal_normal_vert_v2: Field[[ECVDim], wpfloat],
    z_nabla2_e: fa.EKwpField,
    inv_vert_vert_length: fa.EwpField,
    inv_primal_edge_length: fa.EwpField,
    area_edge: fa.EwpField,
    kh_smag_e: fa.EKvpField,
    diff_multfac_vn: fa.KwpField,
    nudgecoeff_e: fa.EwpField,
    vn: fa.EKwpField,
    edge: fa.EintField,
    nudgezone_diff: vpfloat,
    fac_bdydiff_v: wpfloat,
    start_2nd_nudge_line_idx_e: int32,
    limited_area: bool,
) -> fa.EKwpField:
    z_nabla4_e2 = _calculate_nabla4(
        u_vert,
        v_vert,
        primal_normal_vert_v1,
        primal_normal_vert_v2,
        z_nabla2_e,
        inv_vert_vert_length,
        inv_primal_edge_length,
    )

    # TODO: Use if-else statement instead
    vn = (
        where(
            start_2nd_nudge_line_idx_e <= edge,
            _apply_nabla2_and_nabla4_to_vn(
                area_edge,
                kh_smag_e,
                z_nabla2_e,
                z_nabla4_e2,
                diff_multfac_vn,
                nudgecoeff_e,
                vn,
                nudgezone_diff,
            ),
            _apply_nabla2_to_vn_in_lateral_boundary(z_nabla2_e, area_edge, vn, fac_bdydiff_v),
        )
        if limited_area
        else where(
            start_2nd_nudge_line_idx_e <= edge,
            _apply_nabla2_and_nabla4_global_to_vn(
                area_edge,
                kh_smag_e,
                z_nabla2_e,
                z_nabla4_e2,
                diff_multfac_vn,
                vn,
            ),
            vn,
        )
    )

    return vn


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def apply_diffusion_to_vn(
    u_vert: fa.VKvpField,
    v_vert: fa.VKvpField,
    primal_normal_vert_v1: Field[[ECVDim], wpfloat],
    primal_normal_vert_v2: Field[[ECVDim], wpfloat],
    z_nabla2_e: fa.EKwpField,
    inv_vert_vert_length: fa.EwpField,
    inv_primal_edge_length: fa.EwpField,
    area_edge: fa.EwpField,
    kh_smag_e: fa.EKvpField,
    diff_multfac_vn: fa.KwpField,
    nudgecoeff_e: fa.EwpField,
    vn: fa.EKwpField,
    edge: fa.EintField,
    nudgezone_diff: vpfloat,
    fac_bdydiff_v: wpfloat,
    start_2nd_nudge_line_idx_e: int32,
    limited_area: bool,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _apply_diffusion_to_vn(
        u_vert,
        v_vert,
        primal_normal_vert_v1,
        primal_normal_vert_v2,
        z_nabla2_e,
        inv_vert_vert_length,
        inv_primal_edge_length,
        area_edge,
        kh_smag_e,
        diff_multfac_vn,
        nudgecoeff_e,
        vn,
        edge,
        nudgezone_diff,
        fac_bdydiff_v,
        start_2nd_nudge_line_idx_e,
        limited_area,
        out=vn,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
