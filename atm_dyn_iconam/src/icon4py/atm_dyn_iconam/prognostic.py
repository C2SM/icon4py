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

from functional.common import Field

from icon4py.common.dimension import CellDim, EdgeDim, KDim


@dataclass
class PrognosticState:
    """Class that contains the prognostic state.

    corresponds to ICON t_nh_prog
    """

    vertical_wind: Field[[CellDim, KDim], float]  # w(nproma, nlevp1, nblks_c) [m/s]
    normal_wind: Field[[EdgeDim, KDim], float]  # vn(nproma, nlev, nblks_e)  [m/s]
    density: Field[[CellDim, KDim], float]  # rho(nproma, nlev, nblks_c)  [kg/m^3]
    exner_pressure: Field[[CellDim, KDim], float]  # exner(nrpoma, nlev, nblks_c)
    theta_v: Field[[CellDim, KDim], float]  # (nproma, nlev, nlbks_c) [K]
