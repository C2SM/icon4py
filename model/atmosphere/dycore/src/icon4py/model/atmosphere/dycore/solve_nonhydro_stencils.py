# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.atmosphere.dycore.dycore_utils import (
    _broadcast_zero_to_three_edge_kdim_fields_2wp1vp,
)
from icon4py.model.atmosphere.dycore.stencils.init_cell_kdim_field_with_zero_wp import (
    _init_cell_kdim_field_with_zero_vp,
)
from icon4py.model.atmosphere.dycore.stencils.update_density_exner_wind import (
    _update_density_exner_wind,
)
from icon4py.model.atmosphere.dycore.stencils.update_wind import _update_wind
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def init_test_fields(
    z_rho_e: fa.EdgeKField[wpfloat],
    z_theta_v_e: fa.EdgeKField[wpfloat],
    z_dwdz_dd: fa.CellKField[vpfloat],
    z_graddiv_vn: fa.EdgeKField[vpfloat],
    edges_start: gtx.int32,
    edges_end: gtx.int32,
    cells_start: gtx.int32,
    cells_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _broadcast_zero_to_three_edge_kdim_fields_2wp1vp(
        out=(z_rho_e, z_theta_v_e, z_graddiv_vn),
        domain={dims.EdgeDim: (edges_start, edges_end), dims.KDim: (vertical_start, vertical_end)},
    )
    _init_cell_kdim_field_with_zero_vp(
        out=z_dwdz_dd,
        domain={dims.CellDim: (cells_start, cells_end), dims.KDim: (vertical_start, vertical_end)},
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def stencils_61_62(
    rho_now: fa.CellKField[wpfloat],
    grf_tend_rho: fa.CellKField[wpfloat],
    theta_v_now: fa.CellKField[wpfloat],
    grf_tend_thv: fa.CellKField[wpfloat],
    w_now: fa.CellKField[wpfloat],
    grf_tend_w: fa.CellKField[wpfloat],
    rho_new: fa.CellKField[wpfloat],
    exner_new: fa.CellKField[wpfloat],
    w_new: fa.CellKField[wpfloat],
    dtime: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _update_density_exner_wind(
        rho_now,
        grf_tend_rho,
        theta_v_now,
        grf_tend_thv,
        w_now,
        grf_tend_w,
        dtime,
        out=(rho_new, exner_new, w_new),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end - 1),
        },
    )
    _update_wind(
        w_now,
        grf_tend_w,
        dtime,
        out=w_new,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_end - 1, vertical_end),
        },
    )
