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

from icon4py.model.common.dimension import C2CE, C2E, C2EDim, CEDim, CellDim, EdgeDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_divergence_of_flux_of_full3d_graddiv(
    geofac_div: Field[[CEDim], wpfloat],
    z_graddiv_normal: Field[[EdgeDim, KDim], wpfloat],
) -> Field[[CellDim, KDim], vpfloat]:
    z_flxdiv_graddiv_vn_wp = neighbor_sum(geofac_div(C2CE) * z_graddiv_normal(C2E), axis=C2EDim)
    return astype(z_flxdiv_graddiv_vn_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_divergence_of_flux_of_full3d_graddiv(
    geofac_div: Field[[CEDim], wpfloat],
    z_graddiv_normal: Field[[EdgeDim, KDim], vpfloat],
    z_flxdiv_graddiv_vn: Field[[CellDim, KDim], vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_divergence_of_flux_of_full3d_graddiv(
        geofac_div,
        z_graddiv_normal,
        out=z_flxdiv_graddiv_vn,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
