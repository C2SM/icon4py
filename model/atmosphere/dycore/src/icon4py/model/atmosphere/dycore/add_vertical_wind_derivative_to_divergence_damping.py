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
from gt4py.next.ffront.fbuiltins import astype, broadcast

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import E2C, EdgeDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _add_vertical_wind_derivative_to_divergence_damping(
    hmask_dd3d: fa.EdgeField[wpfloat],
    scalfac_dd3d: fa.KField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    z_dwdz_dd: fa.CellKField[vpfloat],
    z_graddiv_vn: fa.EdgeKField[vpfloat],
) -> fa.EdgeKField[vpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_17."""
    z_graddiv_vn_wp = astype(z_graddiv_vn, wpfloat)

    scalfac_dd3d = broadcast(scalfac_dd3d, (EdgeDim, KDim))
    z_graddiv_vn_wp = z_graddiv_vn_wp + (
        hmask_dd3d
        * scalfac_dd3d
        * inv_dual_edge_length
        * astype(z_dwdz_dd(E2C[1]) - z_dwdz_dd(E2C[0]), wpfloat)
    )
    return astype(z_graddiv_vn_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def add_vertical_wind_derivative_to_divergence_damping(
    hmask_dd3d: fa.EdgeField[wpfloat],
    scalfac_dd3d: fa.KField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    z_dwdz_dd: fa.CellKField[vpfloat],
    z_graddiv_vn: fa.EdgeKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _add_vertical_wind_derivative_to_divergence_damping(
        hmask_dd3d,
        scalfac_dd3d,
        inv_dual_edge_length,
        z_dwdz_dd,
        z_graddiv_vn,
        out=z_graddiv_vn,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
