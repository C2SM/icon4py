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
from gt4py.next.ffront.fbuiltins import Field, int32, minimum, maximum

from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _ham_wetdep_compute_critical_radius_ice_phase_ll1(
    zeps_mass  : wpfloat,
    zxtp1c_icnc: Field[[CellDim, KDim], wpfloat]
) -> Field[[CellDim, KDim], bool]:

    ll1_tmp = (zxtp1c_icnc > zeps_mass)

    return ll1_tmp

@field_operator
def _ham_wetdep_compute_critical_radius_ice_phase_ztmp1(
    kmod       : int32,
    zeps       : wpfloat,
    ztmp1      : Field[[CellDim, KDim], wpfloat],
    zxtp1c_icnc: Field[[CellDim, KDim], wpfloat],
    zxtp1c_nas : Field[[CellDim, KDim], wpfloat],
    zxtp1c_ncs : Field[[CellDim, KDim], wpfloat],
    zxtp1c_nks : Field[[CellDim, KDim], wpfloat]
) -> Field[[CellDim, KDim], wpfloat]:

    if (kmod == 4):
        ztmp1 = minimum( wpfloat("1.0"), zxtp1c_icnc / (zxtp1c_ncs + zeps) )

    if (kmod == 3):
        ztmp1 = minimum( wpfloat("1.0"), maximum( wpfloat("0.0"), zxtp1c_icnc - zxtp1c_ncs ) / (zxtp1c_nas + zeps) )

    if (kmod == 2):
        ztmp1 = minimum( wpfloat("1.0"), maximum( wpfloat("0.0"), zxtp1c_icnc - zxtp1c_ncs - zxtp1c_nas ) / (zxtp1c_nks + zeps) )

    return ztmp1


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def ham_wetdep_compute_critical_radius_ice_phase(
    kmod            : int32,
    zeps            : wpfloat,
    zeps_mass       : wpfloat,
    ll1             : Field[[CellDim, KDim], bool],
    ztmp1           : Field[[CellDim, KDim], wpfloat],
    zxtp1c_icnc     : Field[[CellDim, KDim], wpfloat],
    zxtp1c_nas      : Field[[CellDim, KDim], wpfloat],
    zxtp1c_ncs      : Field[[CellDim, KDim], wpfloat],
    zxtp1c_nks      : Field[[CellDim, KDim], wpfloat],
    horizontal_start: int32,
    horizontal_end  : int32,
    vertical_start  : int32,
    vertical_end    : int32
):

    _ham_wetdep_compute_critical_radius_ice_phase_ll1( zeps_mass, zxtp1c_icnc,
                                                       out = ll1,
                                                       domain = {
                                                           CellDim: (horizontal_start, horizontal_end),
                                                           KDim: (vertical_start, vertical_end)
                                                       }
    )

    _ham_wetdep_compute_critical_radius_ice_phase_ztmp1( kmod,
                                                         zeps,
                                                         ztmp1,
                                                         zxtp1c_icnc,
                                                         zxtp1c_nas,
                                                         zxtp1c_ncs,
                                                         zxtp1c_nks,
                                                         out = ztmp1,
                                                         domain = {
                                                             CellDim: (horizontal_start, horizontal_end),
                                                             KDim: (vertical_start, vertical_end)
                                                         }
    )