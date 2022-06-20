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

from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import Field

from icon4py.common.dimension import CellDim, KDim


@field_operator
def _mo_velocity_advection_stencil_12() -> Field[[CellDim, KDim], float]:
    z_w_con_c = float(0.0)
    return z_w_con_c


@program
def mo_velocity_advection_stencil_12(z_w_con_c: Field[[CellDim, KDim], float]):
    _mo_velocity_advection_stencil_12(out=z_w_con_c)
