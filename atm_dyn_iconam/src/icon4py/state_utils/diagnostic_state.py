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

from dataclasses import dataclass

from gt4py.next.common import Field

from icon4py.common.dimension import CellDim, EdgeDim, KDim


@dataclass
class DiagnosticState:
    # fields for 3D elements in turbdiff
    hdef_ic: Field[
        [CellDim, KDim], float
    ]  # ! divergence at half levels(nproma,nlevp1,nblks_c)     [1/s]
    div_ic: Field[
        [CellDim, KDim], float
    ]  # ! horizontal wind field deformation (nproma,nlevp1,nblks_c)     [1/s^2]
    dwdx: Field[
        [CellDim, KDim], float
    ]  # zonal gradient of vertical wind speed (nproma,nlevp1,nblks_c)     [1/s]

    dwdy: Field[
        [CellDim, KDim], float
    ]  # meridional gradient of vertical wind speed (nproma,nlevp1,nblks_c)

    vt: Field[[EdgeDim, KDim], float]
    vn_ie: Field[[EdgeDim, KDim], float]
    w_concorr_c: Field[[CellDim, KDim], float]
    ddt_w_adv_pc: Field[[CellDim, KDim], float]
    ddt_vn_apc_pc: Field[[EdgeDim, KDim], float]
