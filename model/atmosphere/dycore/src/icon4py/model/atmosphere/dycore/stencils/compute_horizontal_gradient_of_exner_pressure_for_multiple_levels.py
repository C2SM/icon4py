# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype
from gt4py.next.experimental import as_offset

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import E2C, E2CDim, Koff
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _compute_horizontal_gradient_of_exner_pressure_for_multiple_levels(
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    temporal_extrapolation_of_perturbed_exner: fa.CellKField[vpfloat],
    zdiff_gradp: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim, dims.KDim], vpfloat],
    ikoffset: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim, dims.KDim], gtx.int32],
    ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: fa.CellKField[vpfloat],
    d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: fa.CellKField[vpfloat],
) -> fa.EdgeKField[vpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_20."""
    z_exner_ex_pr_0 = temporal_extrapolation_of_perturbed_exner(E2C[0])(
        as_offset(Koff, ikoffset[E2CDim(0)])
    )
    z_exner_ex_pr_1 = temporal_extrapolation_of_perturbed_exner(E2C[1])(
        as_offset(Koff, ikoffset[E2CDim(1)])
    )

    z_dexner_dz_c1_0 = ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels(E2C[0])(
        as_offset(Koff, ikoffset[E2CDim(0)])
    )
    z_dexner_dz_c1_1 = ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels(E2C[1])(
        as_offset(Koff, ikoffset[E2CDim(1)])
    )

    z_dexner_dz_c2_0 = d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels(E2C[0])(
        as_offset(Koff, ikoffset[E2CDim(0)])
    )
    z_dexner_dz_c2_1 = d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels(E2C[1])(
        as_offset(Koff, ikoffset[E2CDim(1)])
    )

    z_gradh_exner_wp = inv_dual_edge_length * (
        astype(
            (
                z_exner_ex_pr_1
                + zdiff_gradp[E2CDim(1)]
                * (z_dexner_dz_c1_1 + zdiff_gradp[E2CDim(1)] * z_dexner_dz_c2_1)
            )
            - (
                z_exner_ex_pr_0
                + zdiff_gradp[E2CDim(0)]
                * (z_dexner_dz_c1_0 + zdiff_gradp[E2CDim(0)] * z_dexner_dz_c2_0)
            ),
            wpfloat,
        )
    )

    return astype(z_gradh_exner_wp, vpfloat)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_horizontal_gradient_of_exner_pressure_for_multiple_levels(
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    temporal_extrapolation_of_perturbed_exner: fa.CellKField[vpfloat],
    zdiff_gradp: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim, dims.KDim], vpfloat],
    ikoffset: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim, dims.KDim], gtx.int32],
    ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: fa.CellKField[vpfloat],
    d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: fa.CellKField[vpfloat],
    horizontal_pressure_gradient: fa.EdgeKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_horizontal_gradient_of_exner_pressure_for_multiple_levels(
        inv_dual_edge_length,
        temporal_extrapolation_of_perturbed_exner,
        zdiff_gradp,
        ikoffset,
        ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
        d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
        out=horizontal_pressure_gradient,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
