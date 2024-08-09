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
from gt4py.next.ffront.fbuiltins import Field, int32, neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import E2C2EO, E2C2EODim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _compute_avg_vn(
    e_flx_avg: Field[[dims.EdgeDim, E2C2EODim], wpfloat],
    vn: fa.EdgeKField[wpfloat],
) -> fa.EdgeKField[wpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_31."""
    z_vn_avg_wp = neighbor_sum(e_flx_avg * vn(E2C2EO), axis=E2C2EODim)
    return z_vn_avg_wp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_avg_vn(
    e_flx_avg: Field[[dims.EdgeDim, E2C2EODim], wpfloat],
    vn: fa.EdgeKField[wpfloat],
    z_vn_avg: fa.EdgeKField[wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_avg_vn(
        e_flx_avg,
        vn,
        out=z_vn_avg,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
