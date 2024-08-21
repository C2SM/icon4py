# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import broadcast, int32

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import wpfloat


# TODO: this will have to be removed once domain allows for imports
CellDim = dims.CellDim
KDim = dims.KDim


@field_operator
def _init_cell_kdim_field_with_zero_wp() -> fa.CellKField[wpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_57 or _mo_solve_nonhydro_stencil_64."""
    return broadcast(wpfloat("0.0"), (CellDim, KDim))


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def init_cell_kdim_field_with_zero_wp(
    field_with_zero_wp: fa.CellKField[wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _init_cell_kdim_field_with_zero_wp(
        out=field_with_zero_wp,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
