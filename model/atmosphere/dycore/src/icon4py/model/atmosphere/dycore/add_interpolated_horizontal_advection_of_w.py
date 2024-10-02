# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, astype, int32, neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2CE, C2E, C2EDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _add_interpolated_horizontal_advection_of_w(
    e_bln_c_s: Field[[dims.CEDim], wpfloat],
    z_v_grad_w: fa.EdgeKField[vpfloat],
    ddt_w_adv: fa.CellKField[vpfloat],
) -> fa.CellKField[vpfloat]:
    """Formerly known as _mo_velocity_advection_stencil_17."""
    z_v_grad_w_wp, ddt_w_adv_wp = astype((z_v_grad_w, ddt_w_adv), wpfloat)
    ddt_w_adv_wp = ddt_w_adv_wp + neighbor_sum(z_v_grad_w_wp(C2E) * e_bln_c_s(C2CE), axis=C2EDim)
    return astype(ddt_w_adv_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def add_interpolated_horizontal_advection_of_w(
    e_bln_c_s: Field[[dims.CEDim], wpfloat],
    z_v_grad_w: fa.EdgeKField[vpfloat],
    ddt_w_adv: fa.CellKField[vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _add_interpolated_horizontal_advection_of_w(
        e_bln_c_s,
        z_v_grad_w,
        ddt_w_adv,
        out=ddt_w_adv,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
