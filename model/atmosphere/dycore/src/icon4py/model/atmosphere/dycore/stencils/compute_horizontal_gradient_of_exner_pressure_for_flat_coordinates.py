# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import E2C
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates(
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    temporal_extrapolation_of_perturbed_exner: fa.CellKField[vpfloat],
) -> fa.EdgeKField[vpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_18."""
    z_gradh_exner_wp = inv_dual_edge_length * astype(
        temporal_extrapolation_of_perturbed_exner(E2C[1])
        - temporal_extrapolation_of_perturbed_exner(E2C[0]),
        wpfloat,
    )
    return astype(z_gradh_exner_wp, vpfloat)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates(
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    temporal_extrapolation_of_perturbed_exner: fa.CellKField[vpfloat],
    horizontal_pressure_gradient: fa.EdgeKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates(
        inv_dual_edge_length,
        temporal_extrapolation_of_perturbed_exner,
        out=horizontal_pressure_gradient,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
