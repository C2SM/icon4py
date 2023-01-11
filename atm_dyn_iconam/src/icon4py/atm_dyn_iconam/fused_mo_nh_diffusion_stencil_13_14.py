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
from functional.ffront.fbuiltins import Field
from functional.program_processors.runners import gtfn_cpu

from icon4py.atm_dyn_iconam.mo_nh_diffusion_stencil_13 import (
    _mo_nh_diffusion_stencil_13,
)
from icon4py.atm_dyn_iconam.mo_nh_diffusion_stencil_14 import (
    _mo_nh_diffusion_stencil_14,
)
from icon4py.common.dimension import C2EDim, CellDim, EdgeDim, KDim


@field_operator
def _fused_mo_nh_diffusion_stencil_13_14(
    kh_smag_e: Field[[EdgeDim, KDim], float],
    inv_dual_edge_length: Field[[EdgeDim], float],
    theta_v: Field[[CellDim, KDim], float],
    geofac_div: Field[[CellDim, C2EDim], float],
) -> Field[[CellDim, KDim], float]:
    z_nabla2_e = _mo_nh_diffusion_stencil_13(kh_smag_e, inv_dual_edge_length, theta_v)
    z_temp = _mo_nh_diffusion_stencil_14(z_nabla2_e, geofac_div)
    return z_temp


@program(backend=gtfn_cpu.run_gtfn)
def fused_mo_nh_diffusion_stencil_13_14(
    kh_smag_e: Field[[EdgeDim, KDim], float],
    inv_dual_edge_length: Field[[EdgeDim], float],
    theta_v: Field[[CellDim, KDim], float],
    geofac_div: Field[[CellDim, C2EDim], float],
    z_temp: Field[[CellDim, KDim], float],
    horizontal_start: int,
    horizontal_end: int,
    vertical_start: int,
    vertical_end: int,
):
    _fused_mo_nh_diffusion_stencil_13_14(
        kh_smag_e,
        inv_dual_edge_length,
        theta_v,
        geofac_div,
        out=z_temp,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
