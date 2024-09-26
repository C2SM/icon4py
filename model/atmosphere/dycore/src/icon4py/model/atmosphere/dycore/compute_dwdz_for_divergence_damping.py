# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from gt4py.next import gtx
from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import astype

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_dwdz_for_divergence_damping(
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    w: fa.CellKField[wpfloat],
    w_concorr_c: fa.CellKField[vpfloat],
) -> fa.CellKField[vpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_56_63."""
    inv_ddqz_z_full_wp = astype(inv_ddqz_z_full, wpfloat)

    z_dwdz_dd_wp = inv_ddqz_z_full_wp * (
        (w - w(Koff[1])) - astype(w_concorr_c - w_concorr_c(Koff[1]), wpfloat)
    )
    return astype(z_dwdz_dd_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_dwdz_for_divergence_damping(
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    w: fa.CellKField[wpfloat],
    w_concorr_c: fa.CellKField[vpfloat],
    z_dwdz_dd: fa.CellKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_dwdz_for_divergence_damping(
        inv_ddqz_z_full,
        w,
        w_concorr_c,
        out=z_dwdz_dd,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
