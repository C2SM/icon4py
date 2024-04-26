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
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _ham_wetdep_select_aerosol_wet_radius(
    zeps       : wpfloat,
    zrad_fac   : wpfloat,
    rwet_p_krow: Field[[CellDim, KDim], wpfloat]
) -> (
    tuple[Field[[CellDim, KDim], int32],
          Field[[CellDim, KDim], int32]]
):

    mr = minimum(rwet_p_krow * zrad_fac * wpfloat("1.0e6"), wpfloat("50.0"))

    ll1 = mr > zeps

    ztmp1 = where(ll1, mr, wpfloat("1.0"))
    ztmp2 = floor(wpfloat("3.0") * (log(wpfloat("1.0e4") * ztmp1) / log(wpfloat("2.0"))) + wpfloat("1.0"))

    itmp1 = astype(maximum(wpfloat("0.0"), minimum(wpfloat("60.0"), ztmp2)), int32)
    itmp2 = astype(maximum(wpfloat("0.0"), minimum(wpfloat("60.0"), wpfloat("1.0") + ztmp2)), int32)

    indexy1_phase_mode = where(ll1, itmp1, int32("0"))
    indexy2_phase_mode = where(ll1, itmp2, int32("0"))

    return (indexy1_phase_mode, indexy2_phase_mode)


@program(grid_type=GridType.UNSTRUCTURED)
def ham_wetdep_select_aerosol_wet_radius(
    zeps              : wpfloat,
    zrad_fac          : wpfloat,
    rwet_p_krow       : Field[[CellDim, KDim], wpfloat],
    indexy1_phase_mode: Field[[CellDim, KDim], int32],
    indexy2_phase_mode: Field[[CellDim, KDim], int32],
    horizontal_start  : int32,
    horizontal_end    : int32,
    vertical_start    : int32,
    vertical_end      : int32
):

    _ham_wetdep_select_aerosol_wet_radius( zeps, zrad_fac, rwet_p_krow,
                                           out = (indexy1_phase_mode, indexy2_phase_mode),
                                           domain = {
                                               CellDim: (horizontal_start, horizontal_end),
                                               KDim: (vertical_start, vertical_end)
                                           }
    )