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

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat


@field_operator
def _correct_contravariant_vertical_velocity(
    z_w_con_c: fa.CellKField[vpfloat],
    w_concorr_c: fa.CellKField[vpfloat],
) -> fa.CellKField[vpfloat]:
    """
    Formerly known as _mo_velocity_advection_stencil_13.

    # scidoc:
    # Outputs:
    #  - z_w_con_c :
    #     $$
    #     (\w{\n}{\c}{\k-1/2} - \wcc{\n}{\c}{\k-1/2}) =
    #     \begin{cases}
    #         \w{\n}{\c}{\k-1/2},                        & \k \in [0, \nflatlev+1)     \\
    #         \w{\n}{\c}{\k-1/2} - \wcc{\n}{\c}{\k-1/2}, & \k \in [\nflatlev+1, \nlev) \\
    #         0,                                         & \k = \nlev
    #     \end{cases}
    #     $$
    #     Subtract the contravariant correction $\wcc{}{}{}$ from the
    #     vertical wind $\w{}{}{}$ in the terrain-following levels. This is
    #     done for convevnience here, instead of directly in the advection
    #     tendency update, because the result needs to be interpolated to
    #     edge centers and full levels for later use.
    #     The papers do not use a new symbol for this variable, and the code
    #     ambiguosly mixes the variable names used for
    #     $\wcc{}{}{}$ and $(\w{}{}{} - \wcc{}{}{})$.
    #
    # Inputs:
    #  - $\w{\n}{\c}{\k\pm1/2}$ : w
    #  - $\wcc{\n}{\c}{\k\pm1/2}$ : w_concorr_c
    #
    
    """
    z_w_con_c_vp = z_w_con_c - w_concorr_c
    return z_w_con_c_vp


@program(grid_type=GridType.UNSTRUCTURED)
def correct_contravariant_vertical_velocity(
    w_concorr_c: fa.CellKField[vpfloat],
    z_w_con_c: fa.CellKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _correct_contravariant_vertical_velocity(
        z_w_con_c,
        w_concorr_c,
        out=z_w_con_c,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
