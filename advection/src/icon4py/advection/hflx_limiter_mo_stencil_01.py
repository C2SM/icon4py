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
from functional.ffront.fbuiltins import Field, abs

from icon4py.common.dimension import C2CE, E2C, CEDim, CellDim, EdgeDim, KDim


@field_operator
def _hflx_limiter_mo_stencil_01(
    p_mflx_tracer_h: Field[[EdgeDim, KDim], float],
    p_cc: Field[[CellDim, KDim], float],
    p_mass_flx_e: Field[[EdgeDim, KDim], float],
) -> tuple[Field[[EdgeDim, KDim], float],Field[[EdgeDim, KDim], float]]:

    z_mflx_low = 0.5 * ( p_mass_flx_e * (p_cc(E2C[0]) + p_cc(E2C[1]))
                         - abs(p_mass_flx_e) * (-p_cc(E2C[0]) + p_cc(E2C[1])))

    z_anti = p_mflx_tracer_h - z_mflx_low

    return z_mflx_low, z_anti


@program
def hflx_limiter_mo_stencil_01(
    p_mflx_tracer_h: Field[[EdgeDim, KDim], float],
    p_cc: Field[[CellDim, KDim], float],
    p_mass_flx_e: Field[[EdgeDim, KDim], float],
    z_mflx_low: Field[[EdgeDim, KDim], float],
    z_anti: Field[[EdgeDim, KDim], float],
):
    _hflx_limiter_mo_stencil_01(
        p_mflx_tracer_h,
        p_cc,
        p_mass_flx_e,
        out = (z_mflx_low, z_anti),
    )
