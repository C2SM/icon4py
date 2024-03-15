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
from gt4py.next.ffront.fbuiltins import Field, int32

from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _ham_wetdep_stratiform_case_1(
    ztmst    : wpfloat,
    paclc    : Field[[CellDim, KDim], wpfloat],
    pdpg     : Field[[CellDim, KDim], wpfloat],
    pxtp1c_kt: Field[[CellDim, KDim], wpfloat],
    pxtp10_kt: Field[[CellDim, KDim], wpfloat],
) -> (
    tuple[Field[[CellDim, KDim], wpfloat],
          Field[[CellDim, KDim], wpfloat],
          Field[[CellDim, KDim], wpfloat],
          Field[[CellDim, KDim], wpfloat]]
):

    pxtp1c_kt_tmp = pxtp1c_kt * paclc
    pxtp10_kt_tmp = pxtp10_kt * (1. - paclc)
    zxtp10_tmp    = pxtp10_kt_tmp
    zmf_tmp       = pdpg / ztmst
    
    return (pxtp1c_kt_tmp, pxtp10_kt_tmp, zmf_tmp, zxtp10_tmp)


@program(grid_type=GridType.UNSTRUCTURED)
def ham_wetdep_stratiform_case_1(
    ztmst           : wpfloat,
    paclc           : Field[[CellDim, KDim], wpfloat],
    pdpg            : Field[[CellDim, KDim], wpfloat],
    pxtp1c_kt       : Field[[CellDim, KDim], wpfloat],
    pxtp10_kt       : Field[[CellDim, KDim], wpfloat],
    zmf             : Field[[CellDim, KDim], wpfloat],
    zxtp10          : Field[[CellDim, KDim], wpfloat],
    horizontal_start: int32,
    horizontal_end  : int32,
    vertical_start  : int32,
    vertical_end    : int32
):

    _ham_wetdep_stratiform_case_1( ztmst, paclc, pdpg, pxtp1c_kt, pxtp10_kt,
                                   out = (pxtp1c_kt, pxtp10_kt, zmf, zxtp10),
                                   domain = {
                                       CellDim: (horizontal_start, horizontal_end),
                                       KDim: (vertical_start, vertical_end)
                                   }
    )