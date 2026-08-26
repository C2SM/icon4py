# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _init_louis_scaling_factor(
    cell_area: fa.CellField[wpfloat],
) -> fa.CellField[wpfloat]:
    """
    Compute the scaling factor for the Louis constant b.

    Port of ``compute_scaling_factor_louis`` in ICON's ``mo_tmx_smagorinsky.f90``.
    The scaling factor is designed to be 1 with an R2B8 setup.

    Args:
        cell_area: cell area

    Returns:
        scaling factor for the Louis constant b
    """
    # Global mean cell area of the R2B8 grid [m^2] (``mean_area_R2B8`` in ICON's
    # mo_tmx_smagorinsky.f90). Defined here because module-level closure constants
    # are not supported by the gtfn backend.
    mean_cell_area_r2b8 = wpfloat("97294071.23714285")
    return mean_cell_area_r2b8 / cell_area


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def init_louis_scaling_factor(
    cell_area: fa.CellField[wpfloat],
    scaling_factor_louis: fa.CellField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
) -> None:
    _init_louis_scaling_factor(
        cell_area=cell_area,
        out=scaling_factor_louis,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
        },
    )
