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
from gt4py.next.ffront.fbuiltins import Field, astype

from icon4py.model.common.dimension import EdgeDim, KDim, Koff
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _mo_solve_nonhydro_stencil_36(
    wgtfac_e: Field[[EdgeDim, KDim], vpfloat],
    vn: Field[[EdgeDim, KDim], wpfloat],
    vt: Field[[EdgeDim, KDim], vpfloat],
) -> tuple[
    Field[[EdgeDim, KDim], vpfloat],
    Field[[EdgeDim, KDim], vpfloat],
    Field[[EdgeDim, KDim], vpfloat],
]:
    wgtfac_e_wp, vt_wp = astype((wgtfac_e, vt), wpfloat)

    vn_ie_wp = wgtfac_e_wp * vn + (wpfloat("1.0") - wgtfac_e_wp) * vn(Koff[-1])
    z_vt_ie_wp = astype(wgtfac_e * vt, wpfloat) + (wpfloat("1.0") - wgtfac_e_wp) * vt_wp(Koff[-1])
    z_kin_hor_e_wp = wpfloat("0.5") * (vn**2 + astype(vt**2, wpfloat))
    return astype((vn_ie_wp, z_vt_ie_wp, z_kin_hor_e_wp), vpfloat)


@program(grid_type=GridType.UNSTRUCTURED)
def mo_solve_nonhydro_stencil_36(
    wgtfac_e: Field[[EdgeDim, KDim], vpfloat],
    vn: Field[[EdgeDim, KDim], wpfloat],
    vt: Field[[EdgeDim, KDim], vpfloat],
    vn_ie: Field[[EdgeDim, KDim], vpfloat],
    z_vt_ie: Field[[EdgeDim, KDim], vpfloat],
    z_kin_hor_e: Field[[EdgeDim, KDim], vpfloat],
):
    _mo_solve_nonhydro_stencil_36(
        wgtfac_e, vn, vt, out=(vn_ie[:, 1:], z_vt_ie[:, 1:], z_kin_hor_e[:, 1:])
    )
