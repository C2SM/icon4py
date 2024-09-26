# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from gt4py.next import GridType, gtx
from gt4py.next.ffront.decorator import field_operator, program

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _compute_airmass(
    rho_in: fa.CellKField[wpfloat],
    ddqz_z_full_in: fa.CellKField[wpfloat],
    deepatmo_t1mc_in: fa.KField[wpfloat],
) -> fa.CellKField[wpfloat]:
    return rho_in * ddqz_z_full_in * deepatmo_t1mc_in


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_airmass(
    rho_in: fa.CellKField[wpfloat],
    ddqz_z_full_in: fa.CellKField[wpfloat],
    deepatmo_t1mc_in: fa.KField[wpfloat],
    airmass_out: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_airmass(
        rho_in,
        ddqz_z_full_in,
        deepatmo_t1mc_in,
        out=airmass_out,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
