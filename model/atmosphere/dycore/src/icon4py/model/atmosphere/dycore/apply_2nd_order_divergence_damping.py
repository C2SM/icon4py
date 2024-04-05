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

from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.model_backend import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _apply_2nd_order_divergence_damping(
    z_graddiv_vn: Field[[EdgeDim, KDim], vpfloat],
    vn: Field[[EdgeDim, KDim], wpfloat],
    scal_divdamp_o2: wpfloat,
) -> Field[[EdgeDim, KDim], wpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_26."""
    z_graddiv_vn_wp = astype(z_graddiv_vn, wpfloat)

    vn_wp = vn + (scal_divdamp_o2 * z_graddiv_vn_wp)
    return vn_wp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def apply_2nd_order_divergence_damping(
    z_graddiv_vn: Field[[EdgeDim, KDim], vpfloat],
    vn: Field[[EdgeDim, KDim], wpfloat],
    scal_divdamp_o2: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _apply_2nd_order_divergence_damping(
        z_graddiv_vn,
        vn,
        scal_divdamp_o2,
        out=vn,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
