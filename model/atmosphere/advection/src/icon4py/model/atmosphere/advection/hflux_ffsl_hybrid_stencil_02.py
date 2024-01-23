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
from gt4py.next.ffront.fbuiltins import Field

from icon4py.model.common.dimension import EdgeDim, KDim


@field_operator
def _hflux_ffsl_hybrid_stencil_02(
    p_out_e_hybrid_2: Field[[EdgeDim, KDim], float],
    p_mass_flx_e: Field[[EdgeDim, KDim], float],
    z_dreg_area: Field[[EdgeDim, KDim], float],
) -> Field[[EdgeDim, KDim], float]:
    p_out_e_hybrid_2 = p_mass_flx_e * p_out_e_hybrid_2 / z_dreg_area

    return p_out_e_hybrid_2


@program
def hflux_ffsl_hybrid_stencil_02(
    p_out_e_hybrid_2: Field[[EdgeDim, KDim], float],
    p_mass_flx_e: Field[[EdgeDim, KDim], float],
    z_dreg_area: Field[[EdgeDim, KDim], float],
):
    _hflux_ffsl_hybrid_stencil_02(
        p_out_e_hybrid_2,
        p_mass_flx_e,
        z_dreg_area,
        out=p_out_e_hybrid_2,
    )
