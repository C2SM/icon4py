# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import astype, broadcast, int32

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


# TODO: this will have to be removed once domain allows for imports
EdgeDim = dims.EdgeDim
KDim = dims.KDim


@field_operator
def _apply_weighted_2nd_and_4th_order_divergence_damping(
    scal_divdamp: fa.KField[wpfloat],
    bdy_divdamp: fa.KField[wpfloat],
    nudgecoeff_e: fa.EdgeField[wpfloat],
    z_graddiv2_vn: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
) -> fa.EdgeKField[wpfloat]:
    """Formelry known as _mo_solve_nonhydro_stencil_27."""
    z_graddiv2_vn_wp = astype(z_graddiv2_vn, wpfloat)

    scal_divdamp = broadcast(scal_divdamp, (EdgeDim, KDim))
    bdy_divdamp = broadcast(bdy_divdamp, (EdgeDim, KDim))
    vn_wp = vn + (scal_divdamp + bdy_divdamp * nudgecoeff_e) * z_graddiv2_vn_wp
    return vn_wp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def apply_weighted_2nd_and_4th_order_divergence_damping(
    scal_divdamp: fa.KField[wpfloat],
    bdy_divdamp: fa.KField[wpfloat],
    nudgecoeff_e: fa.EdgeField[wpfloat],
    z_graddiv2_vn: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _apply_weighted_2nd_and_4th_order_divergence_damping(
        scal_divdamp,
        bdy_divdamp,
        nudgecoeff_e,
        z_graddiv2_vn,
        vn,
        out=vn,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
