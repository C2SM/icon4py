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

from icon4py.atm_dyn_iconam.mo_nh_diffusion_stencil_11 import (
    _mo_nh_diffusion_stencil_11,
)
from icon4py.atm_dyn_iconam.mo_nh_diffusion_stencil_12 import (
    _mo_nh_diffusion_stencil_12,
)
from icon4py.common.dimension import CellDim, EdgeDim, KDim


@field_operator
def _fused_mo_nh_diffusion_stencil_11_12(
    theta_v: Field[[CellDim, KDim], float],
    theta_ref_mc: Field[[CellDim, KDim], float],
    thresh_tdiff: float,
    kh_smag_e: Field[[EdgeDim, KDim], float],
) -> Field[[EdgeDim, KDim], float]:
    enh_diffu_3d = _mo_nh_diffusion_stencil_11(theta_v, theta_ref_mc, thresh_tdiff)
    kh_smag_e = _mo_nh_diffusion_stencil_12(kh_smag_e, enh_diffu_3d)
    return kh_smag_e


@program(backend=gtfn_cpu.run_gtfn)
def fused_mo_nh_diffusion_stencil_11_12(
    theta_v: Field[[CellDim, KDim], float],
    theta_ref_mc: Field[[CellDim, KDim], float],
    thresh_tdiff: float,
    kh_smag_e: Field[[EdgeDim, KDim], float],
    horizontal_start: int,
    horizontal_end: int,
    vertical_start: int,
    vertical_end: int,
):
    _fused_mo_nh_diffusion_stencil_11_12(
        theta_v,
        theta_ref_mc,
        thresh_tdiff,
        kh_smag_e,
        out=kh_smag_e,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
