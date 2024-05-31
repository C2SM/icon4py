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
from gt4py.next.ffront.fbuiltins import Field, int32, max_over, maximum
from model.common.tests import field_aliases as fa

from icon4py.model.common.dimension import E2C, E2CDim, EdgeDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat


@field_operator
def _enhance_diffusion_coefficient_for_grid_point_cold_pools(
    kh_smag_e: Field[[EdgeDim, KDim], vpfloat],
    enh_diffu_3d: fa.CKvpField,
) -> Field[[EdgeDim, KDim], vpfloat]:
    kh_smag_e_vp = maximum(kh_smag_e, max_over(enh_diffu_3d(E2C), axis=E2CDim))
    return kh_smag_e_vp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def enhance_diffusion_coefficient_for_grid_point_cold_pools(
    kh_smag_e: Field[[EdgeDim, KDim], vpfloat],
    enh_diffu_3d: fa.CKvpField,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _enhance_diffusion_coefficient_for_grid_point_cold_pools(
        kh_smag_e,
        enh_diffu_3d,
        out=kh_smag_e,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
