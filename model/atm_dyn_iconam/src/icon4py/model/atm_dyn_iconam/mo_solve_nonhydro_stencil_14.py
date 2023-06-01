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
from icon4py.model.common.dimension import EdgeDim, KDim


@field_operator
def _mo_solve_nonhydro_stencil_14() -> (
    tuple[Field[[EdgeDim, KDim], float], Field[[EdgeDim, KDim], float]]
):
    z_rho_e = broadcast(0.0, (EdgeDim, KDim))
    z_theta_v_e = broadcast(0.0, (EdgeDim, KDim))
    return z_rho_e, z_theta_v_e


@program
def mo_solve_nonhydro_stencil_14(
    z_rho_e: Field[[EdgeDim, KDim], float],
    z_theta_v_e: Field[[EdgeDim, KDim], float],
):
    _mo_solve_nonhydro_stencil_14(out=(z_rho_e, z_theta_v_e))
