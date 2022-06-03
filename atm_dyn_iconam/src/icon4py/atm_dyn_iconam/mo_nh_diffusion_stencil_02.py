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
from functional.ffront.fbuiltins import Field, float, neighbor_sum

from icon4py.common.dimension import C2E, C2EDim, CellDim, EdgeDim, KDim


@field_operator
def _mo_nh_diffusion_stencil_02_div(
    vn: Field[[EdgeDim, KDim], float],
    geofac_div: Field[[CellDim, C2EDim], float],
) -> Field[[CellDim, KDim], float]:
    div = neighbor_sum(vn(C2E) * geofac_div, axis=C2EDim)
    return div


@field_operator
def _mo_nh_diffusion_stencil_02_khc(
    kh_smag_ec: Field[[EdgeDim, KDim], float],
    e_bln_c_s: Field[[CellDim, C2EDim], float],
    diff_multfac_smag: Field[[KDim], float],
) -> Field[[CellDim, KDim], float]:
    kh_c = neighbor_sum(kh_smag_ec(C2E) * e_bln_c_s, axis=C2EDim) / diff_multfac_smag
    return kh_c


@program
def mo_nh_diffusion_stencil_02(
    kh_smag_ec: Field[[EdgeDim, KDim], float],
    vn: Field[[EdgeDim, KDim], float],
    e_bln_c_s: Field[[CellDim, C2EDim], float],
    geofac_div: Field[[CellDim, C2EDim], float],
    diff_multfac_smag: Field[[KDim], float],
    kh_c: Field[[CellDim, KDim], float],
    div: Field[[CellDim, KDim], float],
):
    _mo_nh_diffusion_stencil_02_khc(kh_smag_ec, e_bln_c_s, diff_multfac_smag, out=kh_c)
    _mo_nh_diffusion_stencil_02_div(vn, geofac_div, out=div)
