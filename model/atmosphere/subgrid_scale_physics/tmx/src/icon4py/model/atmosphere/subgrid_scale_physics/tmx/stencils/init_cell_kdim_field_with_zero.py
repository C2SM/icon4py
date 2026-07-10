# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import broadcast

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _init_cell_kdim_field_with_zero() -> fa.CellKField[wpfloat]:
    """
    Set a cell K field to zero.

    Port of the ``CALL init(...)`` (mo_fortran_tools) zero fills of the tmx
    scalar diffusion stages (mo_vdf.f90), e.g. the tracer and energy tendencies
    at the top of 'Compute_diffusion_hydrometeors' /
    'Compute_diffusion_temperature'.
    """
    return broadcast(wpfloat("0.0"), (dims.CellDim, dims.KDim))


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def init_cell_kdim_field_with_zero(
    field: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _init_cell_kdim_field_with_zero(
        out=field,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
