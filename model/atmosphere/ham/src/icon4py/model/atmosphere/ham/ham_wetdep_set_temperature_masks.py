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
from gt4py.next.ffront.fbuiltins import Field, int32, astype, maximum, minimum, where, floor, log

from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _ham_wetdep_set_temperature_masks(
    cthomi: wpfloat,
    tmelt : wpfloat,
    ptm1  : Field[[CellDim, KDim], wpfloat]
) -> (
    tuple[Field[[CellDim, KDim], bool],
          Field[[CellDim, KDim], bool],
          Field[[CellDim, KDim], bool]]
):

    ll_wat_tmp = ptm1 > tmelt
    ll_mxp_tmp = (~ ll_wat_tmp) & (ptm1 > cthomi)
    ll_ice_tmp = ptm1 <= cthomi

    return (ll_ice_tmp, ll_mxp_tmp, ll_wat_tmp)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def ham_wetdep_set_temperature_masks(
    cthomi           : wpfloat,
    tmelt            : wpfloat,
    ll_ice           : Field[[CellDim, KDim], bool],
    ll_mxp           : Field[[CellDim, KDim], bool],
    ll_wat           : Field[[CellDim, KDim], bool],
    ptm1             : Field[[CellDim, KDim], wpfloat],
    horizontal_start : int32,
    horizontal_end   : int32,
    vertical_start   : int32,
    vertical_end     : int32
):

    _ham_wetdep_set_temperature_masks( cthomi, tmelt, ptm1,
                                       out = (ll_ice, ll_mxp, ll_wat),
                                       domain = {
                                           CellDim: (horizontal_start, horizontal_end),
                                           KDim: (vertical_start, vertical_end)
                                       }
    )