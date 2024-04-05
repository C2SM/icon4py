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

from icon4py.model.common.dimension import CellDim, KDim, Koff
from icon4py.model.common.model_backend import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_advective_vertical_wind_tendency(
    z_w_con_c: Field[[CellDim, KDim], vpfloat],
    w: Field[[CellDim, KDim], wpfloat],
    coeff1_dwdz: Field[[CellDim, KDim], vpfloat],
    coeff2_dwdz: Field[[CellDim, KDim], vpfloat],
) -> Field[[CellDim, KDim], vpfloat]:
    """Formerly known as _mo_velocity_advection_stencil_16."""
    z_w_con_c_wp = astype(z_w_con_c, wpfloat)
    coeff1_dwdz_wp, coeff2_dwdz_wp = astype((coeff1_dwdz, coeff2_dwdz), wpfloat)

    ddt_w_adv_wp = -z_w_con_c_wp * (
        w(Koff[-1]) * coeff1_dwdz_wp
        - w(Koff[1]) * coeff2_dwdz_wp
        + w * astype(coeff2_dwdz - coeff1_dwdz, wpfloat)
    )
    return astype(ddt_w_adv_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_advective_vertical_wind_tendency(
    z_w_con_c: Field[[CellDim, KDim], vpfloat],
    w: Field[[CellDim, KDim], wpfloat],
    coeff1_dwdz: Field[[CellDim, KDim], vpfloat],
    coeff2_dwdz: Field[[CellDim, KDim], vpfloat],
    ddt_w_adv: Field[[CellDim, KDim], vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_advective_vertical_wind_tendency(
        z_w_con_c,
        w,
        coeff1_dwdz,
        coeff2_dwdz,
        out=ddt_w_adv,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
