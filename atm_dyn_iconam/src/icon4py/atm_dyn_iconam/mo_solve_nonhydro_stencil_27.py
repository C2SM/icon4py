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

from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import Field, broadcast

from icon4py.common.dimension import EdgeDim, KDim


@field_operator
def _mo_solve_nonhydro_stencil_27(
    scal_divdamp: Field[[KDim], float],
    bdy_divdamp: Field[[KDim], float],
    nudgecoeff_e: Field[[EdgeDim], float],
    z_graddiv2_vn: Field[[EdgeDim, KDim], float],
    vn: Field[[EdgeDim, KDim], float],
) -> Field[[EdgeDim, KDim], float]:

    scal_divdamp = broadcast(scal_divdamp, (EdgeDim, KDim))
    bdy_divdamp = broadcast(bdy_divdamp, (EdgeDim, KDim))
    nudgecoeff_e = broadcast(nudgecoeff_e, (EdgeDim, KDim))
    vn = vn + (scal_divdamp + bdy_divdamp * nudgecoeff_e) * z_graddiv2_vn
    return vn


@program
def mo_solve_nonhydro_stencil_27(
    scal_divdamp: Field[[KDim], float],
    bdy_divdamp: Field[[KDim], float],
    nudgecoeff_e: Field[[EdgeDim], float],
    z_graddiv2_vn: Field[[EdgeDim, KDim], float],
    vn: Field[[EdgeDim, KDim], float],
):
    _mo_solve_nonhydro_stencil_27(
        scal_divdamp, bdy_divdamp, nudgecoeff_e, z_graddiv2_vn, vn, out=vn
    )
