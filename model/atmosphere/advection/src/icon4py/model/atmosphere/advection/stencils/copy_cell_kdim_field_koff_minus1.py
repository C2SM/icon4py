# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import CellDim, KDim, Koff


# TODO (dastrm): move this highly generic stencil to common


@field_operator
def _copy_cell_kdim_field_koff_minus1(field_in: fa.CellKField[float]) -> fa.CellKField[float]:
    return field_in(Koff[-1])


@program
def copy_cell_kdim_field_koff_minus1(
    field_in: fa.CellKField[float],
    field_out: fa.CellKField[float],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _copy_cell_kdim_field_koff_minus1(
        field_in,
        out=field_out,
        domain={CellDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)},
    )
