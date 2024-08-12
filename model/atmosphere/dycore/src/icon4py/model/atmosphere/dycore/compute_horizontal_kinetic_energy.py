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
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


# TODO: this will have to be removed once domain allows for imports
EdgeDim = dims.EdgeDim
KDim = dims.KDim


@field_operator
def _compute_horizontal_kinetic_energy(
    vn: fa.EdgeKField[wpfloat],
    vt: fa.EdgeKField[vpfloat],
) -> tuple[
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
]:
    """Formerly known as _mo_solve_nonhydro_stencil_37 or _mo_velocity_advection_stencil_05."""
    vn_ie_wp = vn
    z_vt_ie_vp = vt
    z_kin_hor_e_wp = wpfloat("0.5") * (vn * vn + astype(vt * vt, wpfloat))
    return astype(vn_ie_wp, vpfloat), z_vt_ie_vp, astype(z_kin_hor_e_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_horizontal_kinetic_energy(
    vn: fa.EdgeKField[wpfloat],
    vt: fa.EdgeKField[vpfloat],
    vn_ie: fa.EdgeKField[vpfloat],
    z_vt_ie: fa.EdgeKField[vpfloat],
    z_kin_hor_e: fa.EdgeKField[vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_horizontal_kinetic_energy(
        vn,
        vt,
        out=(vn_ie, z_vt_ie, z_kin_hor_e),
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
