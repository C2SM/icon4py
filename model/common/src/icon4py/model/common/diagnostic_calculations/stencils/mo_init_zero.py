# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, broadcast, int32

from icon4py.model.common.dimension import CellDim, EdgeDim, KDim
from icon4py.model.common.type_alias import vpfloat
from icon4py.model.common.settings import backend


@field_operator
def _mo_init_ddt_cell_zero() -> (
    tuple[
        Field[[CellDim, KDim], vpfloat],
        Field[[CellDim, KDim], vpfloat],
        Field[[CellDim, KDim], vpfloat],
    ]
):
    ddt_exner_phy = broadcast(0.0, (CellDim, KDim))
    ddt_w_adv_ntl1 = broadcast(0.0, (CellDim, KDim))
    ddt_w_adv_ntl2 = broadcast(0.0, (CellDim, KDim))
    return (ddt_exner_phy, ddt_w_adv_ntl1, ddt_w_adv_ntl2)


@field_operator
def _mo_init_ddt_edge_zero() -> (
    tuple[
        Field[[EdgeDim, KDim], vpfloat],
        Field[[EdgeDim, KDim], vpfloat],
        Field[[EdgeDim, KDim], vpfloat],
    ]
):
    ddt_vn_phy = broadcast(0.0, (EdgeDim, KDim))
    ddt_vn_apc_ntl1 = broadcast(0.0, (EdgeDim, KDim))
    ddt_vn_apc_ntl2 = broadcast(0.0, (EdgeDim, KDim))
    return (ddt_vn_phy, ddt_vn_apc_ntl1, ddt_vn_apc_ntl2)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def mo_init_ddt_cell_zero(
    ddt_exner_phy: Field[[CellDim, KDim], vpfloat],
    ddt_w_adv_ntl1: Field[[CellDim, KDim], vpfloat],
    ddt_w_adv_ntl2: Field[[CellDim, KDim], vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _mo_init_ddt_cell_zero(
        out=(ddt_exner_phy, ddt_w_adv_ntl1, ddt_w_adv_ntl2),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def mo_init_ddt_edge_zero(
    ddt_vn_phy: Field[[EdgeDim, KDim], vpfloat],
    ddt_vn_apc_ntl1: Field[[EdgeDim, KDim], vpfloat],
    ddt_vn_apc_ntl2: Field[[EdgeDim, KDim], vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _mo_init_ddt_edge_zero(
        out=(ddt_vn_phy, ddt_vn_apc_ntl1, ddt_vn_apc_ntl2),
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
