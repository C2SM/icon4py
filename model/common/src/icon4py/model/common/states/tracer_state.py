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

from gt4py.next import as_field
from gt4py.next.common import Field

from icon4py.model.common.dimension import C2E2C2EDim, CellDim, KDim


@dataclass
class TracerState:
    """
    Class that contains the tracer state which includes hydrometeors and aerosols.
    Corresponds to tracer pointers in ICON t_nh_prog
    """

    qv: Field[[CellDim, KDim], float]
    # pressure at half levels
    qc: Field[[CellDim, KDim], float]
    qr: Field[[CellDim, KDim], float]
    # zonal wind speed
    qi: Field[[CellDim, KDim], float]
    # meridional wind speed
    qs: Field[[CellDim, KDim], float]
    qg: Field[[CellDim, KDim], float]

