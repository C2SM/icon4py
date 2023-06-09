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
from gt4py.next.ffront.fbuiltins import Field, where

from icon4py.model.common.dimension import EdgeDim, KDim


@field_operator
def _mo_solve_nonhydro_stencil_22(
    ipeidx_dsl: Field[[EdgeDim, KDim], bool],
    pg_exdist: Field[[EdgeDim, KDim], float],
    z_hydro_corr: Field[[EdgeDim], float],
    z_gradh_exner: Field[[EdgeDim, KDim], float],
) -> Field[[EdgeDim, KDim], float]:
    z_gradh_exner = where(
        ipeidx_dsl, z_gradh_exner + z_hydro_corr * pg_exdist, z_gradh_exner
    )
    return z_gradh_exner


@program
def mo_solve_nonhydro_stencil_22(
    ipeidx_dsl: Field[[EdgeDim, KDim], bool],
    pg_exdist: Field[[EdgeDim, KDim], float],
    z_hydro_corr: Field[[EdgeDim], float],
    z_gradh_exner: Field[[EdgeDim, KDim], float],
):
    _mo_solve_nonhydro_stencil_22(
        ipeidx_dsl,
        pg_exdist,
        z_hydro_corr,
        z_gradh_exner,
        out=z_gradh_exner,
    )
