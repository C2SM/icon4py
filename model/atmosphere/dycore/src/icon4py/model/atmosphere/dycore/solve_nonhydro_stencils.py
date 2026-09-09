# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.atmosphere.dycore.stencils.update_density_exner_wind import (
    _update_density_exner_wind,
)
from icon4py.model.atmosphere.dycore.stencils.update_wind import _update_wind
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.math.vertical_operations import (
    _set_constant_on_model_levels_on_cells_vp,
    _set_constant_on_model_levels_on_edges_vp,
    _set_constant_on_model_levels_on_edges_wp,
)
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _init_test_fields() -> tuple[
    fa.EdgeKField[wpfloat],
    fa.EdgeKField[wpfloat],
    fa.EdgeKField[vpfloat],
    fa.CellKField[vpfloat],
]:
    zero_wp = wpfloat(0.0)
    zero_vp = vpfloat(0.0)
    return (
        _set_constant_on_model_levels_on_edges_wp(zero_wp),
        _set_constant_on_model_levels_on_edges_wp(zero_wp),
        _set_constant_on_model_levels_on_edges_vp(zero_vp),
        _set_constant_on_model_levels_on_cells_vp(zero_vp)
    )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def init_test_fields(  # noqa: PLR0917 [too-many-positional-arguments]
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
) -> None:
    _init_test_fields(
        out=(z_rho_e, z_theta_v_e, z_graddiv_vn, z_dwdz_dd),
        domain=({dims.EdgeDim: (edges_start, edges_end), dims.KDim: (vertical_start, vertical_end)},
                {dims.EdgeDim: (edges_start, edges_end), dims.KDim: (vertical_start, vertical_end)},
                {dims.EdgeDim: (edges_start, edges_end), dims.KDim: (vertical_start, vertical_end)},
                {dims.CellDim: (cells_start, cells_end), dims.KDim: (vertical_start, vertical_end)},))


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def stencils_61_62(  # noqa: PLR0917 [too-many-positional-arguments]
    rho_now: fa.CellKField[wpfloat],
    grf_tend_rho: fa.CellKField[wpfloat],
    theta_v_now: fa.CellKField[wpfloat],
    grf_tend_thv: fa.CellKField[wpfloat],
    w_now: fa.CellKHalfField[wpfloat],
    grf_tend_w: fa.CellKHalfField[wpfloat],
    rho_new: fa.CellKField[wpfloat],
    exner_new: fa.CellKField[wpfloat],
    w_new: fa.CellKHalfField[wpfloat],
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
        domain=(
            {
                dims.CellDim: (horizontal_start, horizontal_end),
                dims.KDim: (vertical_start, vertical_end - 1),
            },
            {
                dims.CellDim: (horizontal_start, horizontal_end),
                dims.KDim: (vertical_start, vertical_end - 1),
            },
            {
                dims.CellDim: (horizontal_start, horizontal_end),
                dims.KHalfDim: (vertical_start, vertical_end - 1),
            },
        ),
    )
    _update_wind(
        w_now,
        grf_tend_w,
        dtime,
        out=w_new,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KHalfDim: (vertical_end - 1, vertical_end),
        },
    )
