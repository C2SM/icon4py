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
from gt4py.next.ffront.fbuiltins import Field, broadcast, maximum, minimum, abs

from icon4py.common.dimension import C2CE, C2E, CEDim, E2C, CellDim, EdgeDim, KDim, Koff


@field_operator
def _face_val_ppm_stencil_01a(
    p_cc: Field[[CellDim, KDim], float],
    p_cellhgt_mc_now: Field[[CellDim, KDim], float],
) -> Field[[CellDim, KDim], float]:

    zfac_m1 = (p_cc - p_cc(Koff[-1])) / (p_cellhgt_mc_now + p_cellhgt_mc_now(Koff[-1]))
    zfac = (p_cc(Koff[+1]) - p_cc) / (p_cellhgt_mc_now(Koff[+1]) + p_cellhgt_mc_now)
    z_slope = ( p_cellhgt_mc_now / (p_cellhgt_mc_now(Koff[-1]) + p_cellhgt_mc_now + p_cellhgt_mc_now(Koff[+1])) ) * ( (2.*p_cellhgt_mc_now(Koff[-1]) + p_cellhgt_mc_now) * zfac + (p_cellhgt_mc_now + 2.*p_cellhgt_mc_now(Koff[+1])) * zfac_m1)

    return z_slope


@program
def face_val_ppm_stencil_01a(
    p_cc: Field[[CellDim, KDim], float],
    p_cellhgt_mc_now: Field[[CellDim, KDim], float],
    z_slope: Field[[CellDim, KDim], float],
):
    _face_val_ppm_stencil_01a(
        p_cc,
        p_cellhgt_mc_now,
        out=z_slope,
    )