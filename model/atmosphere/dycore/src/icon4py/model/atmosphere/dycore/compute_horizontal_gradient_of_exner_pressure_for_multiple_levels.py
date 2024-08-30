# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.experimental import as_offset
from gt4py.next.ffront.fbuiltins import Field, astype, int32

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import E2C, E2EC, Koff
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


# TODO: this will have to be removed once domain allows for imports
EdgeDim = dims.EdgeDim
KDim = dims.KDim


@field_operator
def _compute_horizontal_gradient_of_exner_pressure_for_multiple_levels(
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    z_exner_ex_pr: fa.CellKField[vpfloat],
    zdiff_gradp: Field[[dims.ECDim, dims.KDim], vpfloat],
    ikoffset: Field[[dims.ECDim, dims.KDim], int32],
    z_dexner_dz_c_1: fa.CellKField[vpfloat],
    z_dexner_dz_c_2: fa.CellKField[vpfloat],
) -> fa.EdgeKField[vpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_20."""
    z_exner_ex_pr_0 = z_exner_ex_pr(as_offset(Koff, ikoffset(E2EC[0])))
    z_exner_ex_pr_1 = z_exner_ex_pr(as_offset(Koff, ikoffset(E2EC[1])))

    z_dexner_dz_c1_0 = z_dexner_dz_c_1(as_offset(Koff, ikoffset(E2EC[0])))
    z_dexner_dz_c1_1 = z_dexner_dz_c_1(as_offset(Koff, ikoffset(E2EC[1])))

    z_dexner_dz_c2_0 = z_dexner_dz_c_2(as_offset(Koff, ikoffset(E2EC[0])))
    z_dexner_dz_c2_1 = z_dexner_dz_c_2(as_offset(Koff, ikoffset(E2EC[1])))

    z_gradh_exner_wp = inv_dual_edge_length * (
        astype(
            (
                z_exner_ex_pr_1(E2C[1])
                + zdiff_gradp(E2EC[1])
                * (z_dexner_dz_c1_1(E2C[1]) + zdiff_gradp(E2EC[1]) * z_dexner_dz_c2_1(E2C[1]))
            )
            - (
                z_exner_ex_pr_0(E2C[0])
                + zdiff_gradp(E2EC[0])
                * (z_dexner_dz_c1_0(E2C[0]) + zdiff_gradp(E2EC[0]) * z_dexner_dz_c2_0(E2C[0]))
            ),
            wpfloat,
        )
    )

    return astype(z_gradh_exner_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_horizontal_gradient_of_exner_pressure_for_multiple_levels(
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    z_exner_ex_pr: fa.CellKField[vpfloat],
    zdiff_gradp: Field[[dims.ECDim, dims.KDim], vpfloat],
    ikoffset: Field[[dims.ECDim, dims.KDim], int32],
    z_dexner_dz_c_1: fa.CellKField[vpfloat],
    z_dexner_dz_c_2: fa.CellKField[vpfloat],
    z_gradh_exner: fa.EdgeKField[vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_horizontal_gradient_of_exner_pressure_for_multiple_levels(
        inv_dual_edge_length,
        z_exner_ex_pr,
        zdiff_gradp,
        ikoffset,
        z_dexner_dz_c_1,
        z_dexner_dz_c_2,
        out=z_gradh_exner,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
