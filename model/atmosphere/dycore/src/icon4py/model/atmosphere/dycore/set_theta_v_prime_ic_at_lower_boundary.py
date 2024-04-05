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

from icon4py.model.atmosphere.dycore.interpolate_to_surface import _interpolate_to_surface
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.model_backend import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _set_theta_v_prime_ic_at_lower_boundary(
    wgtfacq_c: Field[[CellDim, KDim], vpfloat],
    z_rth_pr: Field[[CellDim, KDim], vpfloat],
    theta_ref_ic: Field[[CellDim, KDim], vpfloat],
) -> tuple[Field[[CellDim, KDim], vpfloat], Field[[CellDim, KDim], wpfloat]]:
    """Formerly known as _mo_solve_nonhydro_stencil_11_upper."""
    z_theta_v_pr_ic_vp = _interpolate_to_surface(wgtfacq_c=wgtfacq_c, interpolant=z_rth_pr)
    theta_v_ic_vp = theta_ref_ic + z_theta_v_pr_ic_vp
    return z_theta_v_pr_ic_vp, astype(theta_v_ic_vp, wpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def set_theta_v_prime_ic_at_lower_boundary(
    wgtfacq_c: Field[[CellDim, KDim], vpfloat],
    z_rth_pr: Field[[CellDim, KDim], vpfloat],
    theta_ref_ic: Field[[CellDim, KDim], vpfloat],
    z_theta_v_pr_ic: Field[[CellDim, KDim], vpfloat],
    theta_v_ic: Field[[CellDim, KDim], wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _set_theta_v_prime_ic_at_lower_boundary(
        wgtfacq_c,
        z_rth_pr,
        theta_ref_ic,
        out=(z_theta_v_pr_ic, theta_v_ic),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
