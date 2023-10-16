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

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, max_over, maximum

from icon4py.model.common.dimension import E2C, CellDim, E2CDim, EdgeDim, KDim
from icon4py.model.common.type_alias import vpfloat


@field_operator
def _enhance_diffusion_coefficient_for_grid_point_cold_pools(
    kh_smag_e: Field[[EdgeDim, KDim], vpfloat],
    enh_diffu_3d: Field[[CellDim, KDim], vpfloat],
) -> Field[[EdgeDim, KDim], vpfloat]:
    kh_smag_e_vp = maximum(kh_smag_e, max_over(enh_diffu_3d(E2C), axis=E2CDim))
    return kh_smag_e_vp


@program(grid_type=GridType.UNSTRUCTURED)
def enhance_diffusion_coefficient_for_grid_point_cold_pools(
    kh_smag_e: Field[[EdgeDim, KDim], vpfloat],
    enh_diffu_3d: Field[[CellDim, KDim], vpfloat],
):
    _enhance_diffusion_coefficient_for_grid_point_cold_pools(kh_smag_e, enh_diffu_3d, out=kh_smag_e)
