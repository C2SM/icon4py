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

from icon4py.model.common.dimension import CellDim, KDim, C2E2C2EDim


@dataclass
class DiagnosticState:
    """Class that contains the diagnostic state which is not used in dycore but may be used in physics.
    These variables are only stored for output purpose.

    Corresponds to ICON t_nh_diag
    """

    pressure: Field[[CellDim, KDim], float]
    pressure_ifc: Field[[CellDim, KDim], float] # has the same K dimension size with full-level variables because surface pressure is defined separately
    temperature: Field[[CellDim, KDim], float]
    pressure_sfc: Field[[CellDim], float]
    u: Field[[CellDim, KDim], float]
    v: Field[[CellDim, KDim], float]

@dataclass
class DiagnosticMetricState:
    """Class that contains the diagnostic metric state for computing the diagnostic state.

    """

    ddqz_z_full: Field[[CellDim, KDim], float]
    rbf_vec_coeff_c1: Field[[CellDim, C2E2C2EDim], float]
    rbf_vec_coeff_c2: Field[[CellDim, C2E2C2EDim], float]
