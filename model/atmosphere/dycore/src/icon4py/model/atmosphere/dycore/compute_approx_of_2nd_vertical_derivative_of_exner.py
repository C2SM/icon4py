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
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import CellDim, KDim, Koff
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat


@field_operator
def _compute_approx_of_2nd_vertical_derivative_of_exner(
    z_theta_v_pr_ic: fa.CellKField[vpfloat],
    d2dexdz2_fac1_mc: fa.CellKField[vpfloat],
    d2dexdz2_fac2_mc: fa.CellKField[vpfloat],
    z_rth_pr_2: fa.CellKField[vpfloat],
) -> fa.CellKField[vpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_12."""
    z_dexner_dz_c_2_vp = -vpfloat("0.5") * (
        (z_theta_v_pr_ic - z_theta_v_pr_ic(Koff[1])) * d2dexdz2_fac1_mc
        + z_rth_pr_2 * d2dexdz2_fac2_mc
    )
    return z_dexner_dz_c_2_vp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_approx_of_2nd_vertical_derivative_of_exner(
    z_theta_v_pr_ic: fa.CellKField[vpfloat],
    d2dexdz2_fac1_mc: fa.CellKField[vpfloat],
    d2dexdz2_fac2_mc: fa.CellKField[vpfloat],
    z_rth_pr_2: fa.CellKField[vpfloat],
    z_dexner_dz_c_2: fa.CellKField[vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_approx_of_2nd_vertical_derivative_of_exner(
        z_theta_v_pr_ic,
        d2dexdz2_fac1_mc,
        d2dexdz2_fac2_mc,
        z_rth_pr_2,
        out=z_dexner_dz_c_2,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
