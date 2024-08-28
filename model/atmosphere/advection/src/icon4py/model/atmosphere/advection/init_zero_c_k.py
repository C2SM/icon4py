# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import broadcast

from icon4py.model.common import dimension as dims, field_type_aliases as fa


# TODO: this will have to be removed once domain allows for imports
CellDim = dims.CellDim
KDim = dims.KDim


@field_operator
def _init_zero_c_k() -> fa.CellKField[float]:
    return broadcast(0.0, (CellDim, KDim))


@program(grid_type=GridType.UNSTRUCTURED)
def init_zero_c_k(field: fa.CellKField[float]):
    _init_zero_c_k(out=field)
