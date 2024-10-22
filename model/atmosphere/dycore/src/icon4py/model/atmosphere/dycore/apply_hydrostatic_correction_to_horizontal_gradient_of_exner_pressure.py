# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat


@field_operator
def _apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure(
    ipeidx_dsl: fa.EdgeKField[bool],
    pg_exdist: fa.EdgeKField[vpfloat],
    z_hydro_corr: gtx.Field[gtx.Dims[dims.EdgeDim], vpfloat],
    z_gradh_exner: fa.EdgeKField[vpfloat],
) -> fa.EdgeKField[vpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_22."""
    z_gradh_exner_vp = where(ipeidx_dsl, z_gradh_exner + z_hydro_corr * pg_exdist, z_gradh_exner)
    return z_gradh_exner_vp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure(
    ipeidx_dsl: fa.EdgeKField[bool],
    pg_exdist: fa.EdgeKField[vpfloat],
    z_hydro_corr: gtx.Field[gtx.Dims[dims.EdgeDim], vpfloat],
    z_gradh_exner: fa.EdgeKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure(
        ipeidx_dsl,
        pg_exdist,
        z_hydro_corr,
        z_gradh_exner,
        out=z_gradh_exner,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
