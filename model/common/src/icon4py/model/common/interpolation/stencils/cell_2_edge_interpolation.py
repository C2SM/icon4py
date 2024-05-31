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
from gt4py.next.ffront.fbuiltins import Field, int32, neighbor_sum
from model.common.tests import field_type_aliases as fa

from icon4py.model.common.dimension import E2C, E2CDim, EdgeDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _cell_2_edge_interpolation(
    in_field: fa.CKvpField, coeff: Field[[EdgeDim, E2CDim], vpfloat]
) -> fa.EKvpField:
    """
    Interpolate a Cell Field to Edges.

    There is a special handling of lateral boundary edges in `subroutine cells2edges_scalar`
    in mo_icon_interpolation.f90 where the value is set to the one valid in_field value without
    multiplication by coeff. This essentially means: the skip value neighbor in the neighbor_sum
    is skipped and coeff needs to be 1 for this Edge index.
    """
    return neighbor_sum(in_field(E2C) * coeff, axis=E2CDim)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def cell_2_edge_interpolation(
    in_field: fa.CKvpField,
    coeff: Field[[EdgeDim, E2CDim], wpfloat],
    out_field: fa.EKvpField,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _cell_2_edge_interpolation(
        in_field,
        coeff,
        out=out_field,
        domain={EdgeDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)},
    )
