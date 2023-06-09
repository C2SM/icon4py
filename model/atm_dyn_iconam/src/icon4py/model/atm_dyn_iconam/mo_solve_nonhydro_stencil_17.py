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

from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, broadcast

from icon4py.model.common.dimension import E2C, CellDim, EdgeDim, KDim


@field_operator
def _mo_solve_nonhydro_stencil_17(
    hmask_dd3d: Field[[EdgeDim], float],
    scalfac_dd3d: Field[[KDim], float],
    inv_dual_edge_length: Field[[EdgeDim], float],
    z_dwdz_dd: Field[[CellDim, KDim], float],
    z_graddiv_vn: Field[[EdgeDim, KDim], float],
) -> Field[[EdgeDim, KDim], float]:
    scalfac_dd3d = broadcast(scalfac_dd3d, (EdgeDim, KDim))
    z_graddiv_vn = z_graddiv_vn + (
        hmask_dd3d * scalfac_dd3d * inv_dual_edge_length * (z_dwdz_dd(E2C[1]) - z_dwdz_dd(E2C[0]))
    )
    return z_graddiv_vn


@program
def mo_solve_nonhydro_stencil_17(
    hmask_dd3d: Field[[EdgeDim], float],
    scalfac_dd3d: Field[[KDim], float],
    inv_dual_edge_length: Field[[EdgeDim], float],
    z_dwdz_dd: Field[[CellDim, KDim], float],
    z_graddiv_vn: Field[[EdgeDim, KDim], float],
):
    _mo_solve_nonhydro_stencil_17(
        hmask_dd3d,
        scalfac_dd3d,
        inv_dual_edge_length,
        z_dwdz_dd,
        z_graddiv_vn,
        out=z_graddiv_vn,
    )
