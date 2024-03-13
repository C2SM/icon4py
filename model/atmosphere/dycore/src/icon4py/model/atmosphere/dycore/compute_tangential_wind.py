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
from gt4py.next.ffront.fbuiltins import Field, astype, int32, neighbor_sum

from icon4py.model.common.dimension import E2C2E, E2C2EDim, EdgeDim, KDim
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.model_backend import backend


@field_operator
def _compute_tangential_wind(
    vn: Field[[EdgeDim, KDim], wpfloat],
    rbf_vec_coeff_e: Field[[EdgeDim, E2C2EDim], wpfloat],
) -> Field[[EdgeDim, KDim], vpfloat]:
    """Formerly knowan as _mo_velocity_advection_stencil_01."""
    vt_wp = neighbor_sum(rbf_vec_coeff_e * vn(E2C2E), axis=E2C2EDim)
    return astype(vt_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_tangential_wind(
    vn: Field[[EdgeDim, KDim], wpfloat],
    rbf_vec_coeff_e: Field[[EdgeDim, E2C2EDim], wpfloat],
    vt: Field[[EdgeDim, KDim], vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_tangential_wind(
        vn,
        rbf_vec_coeff_e,
        out=vt,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
