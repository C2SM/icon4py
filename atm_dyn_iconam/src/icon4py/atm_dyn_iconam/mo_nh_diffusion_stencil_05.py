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
from functional.ffront.fbuiltins import Field, maximum

from icon4py.common.dimension import EdgeDim, KDim


@field_operator
def _mo_nh_diffusion_stencil_05_global_mode(
    area_edge: Field[[EdgeDim], float],
    kh_smag_e: Field[[EdgeDim, KDim], float],
    z_nabla2_e: Field[[EdgeDim, KDim], float],
    z_nabla_e2: Field[[EdgeDim, KDim], float],
    diff_multfac_vn: Field[[KDim], float],
    vn: Field[[EdgeDim, KDim], float],
) -> Field[[EdgeDim, KDim], float]:
    z_d_vn_hdf = area_edge * (
        kh_smag_e * z_nabla2_e - diff_multfac_vn * z_nabla_e2 * area_edge
    )
    return vn + z_d_vn_hdf


@program
def mo_nh_diffusion_stencil_05_global_mode(
    area_edge: Field[[EdgeDim], float],
    kh_smag_e: Field[[EdgeDim, KDim], float],
    z_nabla2_e: Field[[EdgeDim, KDim], float],
    z_nabla4_e2: Field[[EdgeDim, KDim], float],
    diff_multfac_vn: Field[[KDim], float],
    vn: Field[[EdgeDim, KDim], float],
    horizontal_start: int,
    horizontal_end: int,
    vertical_start: int,
    vertical_end: int,
):
    _mo_nh_diffusion_stencil_05_global_mode(
        area_edge,
        kh_smag_e,
        z_nabla2_e,
        z_nabla4_e2,
        diff_multfac_vn,
        vn,
        out=vn,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@field_operator
def _mo_nh_diffusion_stencil_05(
    area_edge: Field[[EdgeDim], float],
    kh_smag_e: Field[[EdgeDim, KDim], float],
    z_nabla2_e: Field[[EdgeDim, KDim], float],
    z_nabla4_e2: Field[[EdgeDim, KDim], float],
    diff_multfac_vn: Field[[KDim], float],
    nudgecoeff_e: Field[[EdgeDim], float],
    vn: Field[[EdgeDim, KDim], float],
    nudgezone_diff: float,
) -> Field[[EdgeDim, KDim], float]:
    vn = vn + area_edge * (
        maximum(nudgezone_diff * nudgecoeff_e, kh_smag_e) * z_nabla2_e
        - diff_multfac_vn * z_nabla4_e2 * area_edge
    )
    return vn


@program
def mo_nh_diffusion_stencil_05(
    area_edge: Field[[EdgeDim], float],
    kh_smag_e: Field[[EdgeDim, KDim], float],
    z_nabla2_e: Field[[EdgeDim, KDim], float],
    z_nabla4_e2: Field[[EdgeDim, KDim], float],
    diff_multfac_vn: Field[[KDim], float],
    nudgecoeff_e: Field[[EdgeDim], float],
    vn: Field[[EdgeDim, KDim], float],
    nudgezone_diff: float,
):
    _mo_nh_diffusion_stencil_05(
        area_edge,
        kh_smag_e,
        z_nabla2_e,
        z_nabla4_e2,
        diff_multfac_vn,
        nudgecoeff_e,
        vn,
        nudgezone_diff,
        out=vn,
    )
