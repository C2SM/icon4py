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

from icon4py.model.common.dimension import E2C2E, E2C2EO, E2C2EDim, E2C2EODim, EdgeDim, KDim


@field_operator
def _mo_solve_nonhydro_stencil_30(
    e_flx_avg: Field[[EdgeDim, E2C2EODim], float],
    vn: Field[[EdgeDim, KDim], float],
    geofac_grdiv: Field[[EdgeDim, E2C2EODim], float],
    rbf_vec_coeff_e: Field[[EdgeDim, E2C2EDim], float],
) -> tuple[
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
    Field[[EdgeDim, KDim], float],
]:
    z_vn_avg = neighbor_sum(e_flx_avg * vn(E2C2EO), axis=E2C2EODim)
    z_graddiv_vn = neighbor_sum(geofac_grdiv * vn(E2C2EO), axis=E2C2EODim)
    vt = neighbor_sum(rbf_vec_coeff_e * vn(E2C2E), axis=E2C2EDim)
    return z_vn_avg, z_graddiv_vn, vt


@program(grid_type=GridType.UNSTRUCTURED)
def mo_solve_nonhydro_stencil_30(
    e_flx_avg: Field[[EdgeDim, E2C2EODim], float],
    vn: Field[[EdgeDim, KDim], float],
    geofac_grdiv: Field[[EdgeDim, E2C2EODim], float],
    rbf_vec_coeff_e: Field[[EdgeDim, E2C2EDim], float],
    z_vn_avg: Field[[EdgeDim, KDim], float],
    z_graddiv_vn: Field[[EdgeDim, KDim], float],
    vt: Field[[EdgeDim, KDim], float],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _mo_solve_nonhydro_stencil_30(
        e_flx_avg,
        vn,
        geofac_grdiv,
        rbf_vec_coeff_e,
        out=(z_vn_avg, z_graddiv_vn, vt),
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
