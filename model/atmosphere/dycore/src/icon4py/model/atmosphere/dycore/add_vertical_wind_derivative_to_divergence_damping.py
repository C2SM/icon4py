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
from gt4py.next.ffront.fbuiltins import Field, astype, broadcast, int32

from icon4py.model.common.dimension import E2C, CellDim, EdgeDim, KDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _add_vertical_wind_derivative_to_divergence_damping(
    hmask_dd3d: Field[[EdgeDim], wpfloat],
    scalfac_dd3d: Field[[KDim], wpfloat],
    inv_dual_edge_length: Field[[EdgeDim], wpfloat],
    z_dwdz_dd: Field[[CellDim, KDim], vpfloat],
    z_graddiv_vn: Field[[EdgeDim, KDim], vpfloat],
) -> Field[[EdgeDim, KDim], vpfloat]:
    '''Formerly known as _mo_solve_nonhydro_stencil_17.'''
    z_graddiv_vn_wp = astype(z_graddiv_vn, wpfloat)

    scalfac_dd3d = broadcast(scalfac_dd3d, (EdgeDim, KDim))
    z_graddiv_vn_wp = z_graddiv_vn_wp + (
        hmask_dd3d
        * scalfac_dd3d
        * inv_dual_edge_length
        * astype(z_dwdz_dd(E2C[1]) - z_dwdz_dd(E2C[0]), wpfloat)
    )
    return astype(z_graddiv_vn_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED)
def add_vertical_wind_derivative_to_divergence_damping(
    hmask_dd3d: Field[[EdgeDim], wpfloat],
    scalfac_dd3d: Field[[KDim], wpfloat],
    inv_dual_edge_length: Field[[EdgeDim], wpfloat],
    z_dwdz_dd: Field[[CellDim, KDim], vpfloat],
    z_graddiv_vn: Field[[EdgeDim, KDim], vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _add_vertical_wind_derivative_to_divergence_damping(
        hmask_dd3d,
        scalfac_dd3d,
        inv_dual_edge_length,
        z_dwdz_dd,
        z_graddiv_vn,
        out=z_graddiv_vn,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
