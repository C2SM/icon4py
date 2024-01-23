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

from gt4py.next.common import Field, GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import broadcast

from icon4py.model.common.dimension import CellDim


@field_operator
def _set_zero_c() -> Field[[CellDim], float]:
    return broadcast(0.0, (CellDim,))


@program(grid_type=GridType.UNSTRUCTURED)
def set_zero_c(field: Field[[CellDim], float]):
    _set_zero_c(out=field)
