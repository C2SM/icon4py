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

from functional.common import Field
from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import broadcast

from icon4py.common.dimension import CellDim, KDim


@field_operator
def _set_zero() -> Field[[CellDim, KDim], float]:
    return broadcast(0.0, (CellDim, KDim))


@program
def set_zero(field: Field[[CellDim, KDim], float]):
    _set_zero(out=field)
