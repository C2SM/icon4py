# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, astype, int32, neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2CE, C2E, C2EDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


# TODO: this will have to be removed once domain allows for imports
CellDim = dims.CellDim
KDim = dims.KDim


@field_operator
def _interpolate_to_cell_center(
    interpolant: fa.EdgeKField[vpfloat],
    e_bln_c_s: Field[[dims.CEDim], wpfloat],
) -> fa.CellKField[vpfloat]:
    """Formerly known as mo_velocity_advection_stencil_08 or mo_velocity_advection_stencil_09."""
    interpolant_wp = astype(interpolant, wpfloat)
    interpolation_wp = neighbor_sum(e_bln_c_s(C2CE) * interpolant_wp(C2E), axis=C2EDim)
    return astype(interpolation_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def interpolate_to_cell_center(
    interpolant: fa.EdgeKField[vpfloat],
    e_bln_c_s: Field[[dims.CEDim], wpfloat],
    interpolation: fa.CellKField[vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _interpolate_to_cell_center(
        interpolant,
        e_bln_c_s,
        out=interpolation,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
