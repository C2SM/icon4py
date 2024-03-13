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
from gt4py.next.ffront.fbuiltins import Field, astype, broadcast, int32

from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.model_backend import backend


@field_operator
def _apply_weighted_2nd_and_4th_order_divergence_damping(
    scal_divdamp: Field[[KDim], wpfloat],
    bdy_divdamp: Field[[KDim], wpfloat],
    nudgecoeff_e: Field[[EdgeDim], wpfloat],
    z_graddiv2_vn: Field[[EdgeDim, KDim], vpfloat],
    vn: Field[[EdgeDim, KDim], wpfloat],
) -> Field[[EdgeDim, KDim], wpfloat]:
    """Formelry known as _mo_solve_nonhydro_stencil_27."""
    z_graddiv2_vn_wp = astype(z_graddiv2_vn, wpfloat)

    scal_divdamp = broadcast(scal_divdamp, (EdgeDim, KDim))
    bdy_divdamp = broadcast(bdy_divdamp, (EdgeDim, KDim))
    vn_wp = vn + (scal_divdamp + bdy_divdamp * nudgecoeff_e) * z_graddiv2_vn_wp
    return vn_wp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def apply_weighted_2nd_and_4th_order_divergence_damping(
    scal_divdamp: Field[[KDim], wpfloat],
    bdy_divdamp: Field[[KDim], wpfloat],
    nudgecoeff_e: Field[[EdgeDim], wpfloat],
    z_graddiv2_vn: Field[[EdgeDim, KDim], vpfloat],
    vn: Field[[EdgeDim, KDim], wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _apply_weighted_2nd_and_4th_order_divergence_damping(
        scal_divdamp,
        bdy_divdamp,
        nudgecoeff_e,
        z_graddiv2_vn,
        vn,
        out=vn,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
