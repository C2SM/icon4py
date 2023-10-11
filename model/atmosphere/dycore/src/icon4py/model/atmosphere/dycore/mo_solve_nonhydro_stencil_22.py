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
from gt4py.next.ffront.decorator import field_operator, program, scan_operator
from gt4py.next.ffront.fbuiltins import Field, int32, where

from icon4py.model.common.dimension import EdgeDim, KDim


@scan_operator(axis=KDim, forward=False, init=0.0)
def _z_hydro_corr_22_scan(state: float, z_hydro_corr: float) -> float:
    return state + z_hydro_corr


@field_operator
def _z_hydro_corr_22(
    z_hydro_corr: Field[[EdgeDim, KDim], float],
    vert_idx: Field[[EdgeDim, KDim], int32],
) -> Field[[EdgeDim, KDim], float]:
    z_hydro_corr = where(vert_idx == 64, z_hydro_corr, 0.0)
    z_hydro_corr_horizontal = _z_hydro_corr_22_scan(z_hydro_corr)
    return z_hydro_corr_horizontal


@field_operator
def _mo_solve_nonhydro_stencil_22(
    ipeidx_dsl: Field[[EdgeDim, KDim], bool],
    pg_exdist: Field[[EdgeDim, KDim], float],
    # z_hydro_corr: Field[[EdgeDim], float],
    z_hydro_corr: Field[[EdgeDim, KDim], float],
    z_gradh_exner: Field[[EdgeDim, KDim], float],
) -> Field[[EdgeDim, KDim], float]:
    z_gradh_exner = where(ipeidx_dsl, z_gradh_exner + z_hydro_corr * pg_exdist, z_gradh_exner)
    return z_gradh_exner


@program(grid_type=GridType.UNSTRUCTURED)
def mo_solve_nonhydro_stencil_22(
    ipeidx_dsl: Field[[EdgeDim, KDim], bool],
    pg_exdist: Field[[EdgeDim, KDim], float],
    # z_hydro_corr: Field[[EdgeDim], float],
    z_hydro_corr: Field[[EdgeDim, KDim], float],
    z_gradh_exner: Field[[EdgeDim, KDim], float],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _mo_solve_nonhydro_stencil_22(
        ipeidx_dsl,
        pg_exdist,
        z_hydro_corr,
        z_gradh_exner,
        out=z_gradh_exner,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
