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
from functional.ffront.fbuiltins import Field, max_over, maximum

from icon4py.common.dimension import E2C, CellDim, E2CDim, EdgeDim, KDim


@field_operator
def _mo_nh_diffusion_stencil_12(
    kh_smag_e: Field[[EdgeDim, KDim], float],
    enh_diffu_3d: Field[[CellDim, KDim], float],
) -> Field[[EdgeDim, KDim], float]:
    kh_smag_e = maximum(kh_smag_e, max_over(enh_diffu_3d(E2C), axis=E2CDim))
    return kh_smag_e


@program
def mo_nh_diffusion_stencil_12(
    kh_smag_e: Field[[EdgeDim, KDim], float],
    enh_diffu_3d: Field[[CellDim, KDim], float],
):
    _mo_nh_diffusion_stencil_12(kh_smag_e, enh_diffu_3d, out=kh_smag_e)
