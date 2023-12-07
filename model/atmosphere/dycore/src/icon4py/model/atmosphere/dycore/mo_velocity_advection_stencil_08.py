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
from gt4py.next.ffront.fbuiltins import Field, astype, int32, neighbor_sum

from icon4py.model.common.dimension import C2CE, C2E, C2EDim, CEDim, CellDim, EdgeDim, KDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _mo_velocity_advection_stencil_08(
    z_kin_hor_e: Field[[EdgeDim, KDim], vpfloat],
    e_bln_c_s: Field[[CEDim], wpfloat],
) -> Field[[CellDim, KDim], vpfloat]:
    z_kin_hor_e_wp = astype(z_kin_hor_e, wpfloat)

    z_ekinh_wp = neighbor_sum(z_kin_hor_e_wp(C2E) * e_bln_c_s(C2CE), axis=C2EDim)
    return astype(z_ekinh_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED)
def mo_velocity_advection_stencil_08(
    z_kin_hor_e: Field[[EdgeDim, KDim], vpfloat],
    e_bln_c_s: Field[[CEDim], wpfloat],
    z_ekinh: Field[[CellDim, KDim], vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _mo_velocity_advection_stencil_08(
        z_kin_hor_e,
        e_bln_c_s,
        out=z_ekinh,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
