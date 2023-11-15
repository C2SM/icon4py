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
from gt4py.next.ffront.fbuiltins import Field

from icon4py.model.common.dimension import EdgeDim, KDim, Koff


@field_operator
def _mo_solve_nonhydro_stencil_36(
    wgtfac_e: Field[[EdgeDim, KDim], float],
    vn: Field[[EdgeDim, KDim], float],
    vt: Field[[EdgeDim, KDim], float],
) -> tuple[
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
]:
    vn_ie = wgtfac_e * vn + (1.0 - wgtfac_e) * vn(Koff[-1])
    z_vt_ie = wgtfac_e * vt + (1.0 - wgtfac_e) * vt(Koff[-1])
    # TODO(magdalena): change exponent back to int (workaround for gt4py)
    z_kin_hor_e = 0.5 * (vn*vn + vt*vt)
    return vn_ie, z_vt_ie, z_kin_hor_e


@program(grid_type=GridType.UNSTRUCTURED)
def mo_solve_nonhydro_stencil_36(
    wgtfac_e: Field[[EdgeDim, KDim], float],
    vn: Field[[EdgeDim, KDim], float],
    vt: Field[[EdgeDim, KDim], float],
    vn_ie: Field[[EdgeDim, KDim], float],
    z_vt_ie: Field[[EdgeDim, KDim], float],
    z_kin_hor_e: Field[[EdgeDim, KDim], float],
):
    _mo_solve_nonhydro_stencil_36(
        wgtfac_e, vn, vt, out=(vn_ie[:, 1:], z_vt_ie[:, 1:], z_kin_hor_e[:, 1:])
    )
