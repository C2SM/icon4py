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
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_contravariant_correction(
    vn: Field[[EdgeDim, KDim], wpfloat],
    ddxn_z_full: Field[[EdgeDim, KDim], vpfloat],
    ddxt_z_full: Field[[EdgeDim, KDim], vpfloat],
    vt: Field[[EdgeDim, KDim], vpfloat],
) -> Field[[EdgeDim, KDim], vpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_35 or mo_velocity_advection_stencil_04."""
    ddxn_z_full_wp = astype(ddxn_z_full, wpfloat)

    z_w_concorr_me_wp = vn * ddxn_z_full_wp + astype(vt * ddxt_z_full, wpfloat)
    return astype(z_w_concorr_me_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_contravariant_correction(
    vn: Field[[EdgeDim, KDim], wpfloat],
    ddxn_z_full: Field[[EdgeDim, KDim], vpfloat],
    ddxt_z_full: Field[[EdgeDim, KDim], vpfloat],
    vt: Field[[EdgeDim, KDim], vpfloat],
    z_w_concorr_me: Field[[EdgeDim, KDim], vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_contravariant_correction(
        vn,
        ddxn_z_full,
        ddxt_z_full,
        vt,
        out=z_w_concorr_me,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
