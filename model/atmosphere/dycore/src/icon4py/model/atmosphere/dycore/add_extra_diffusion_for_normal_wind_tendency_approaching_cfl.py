# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import (
    Field,
    abs,
    astype,
    int32,
    minimum,
    neighbor_sum,
    where,
)

from icon4py.model.atmosphere.dycore.init_two_edge_kdim_fields_with_zero_wp import (
    _init_two_edge_kdim_fields_with_zero_wp,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import (
    E2C,
    E2C2EO,
    E2V,
    E2C2EODim,
    E2CDim,
    Koff,
)
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _add_extra_diffusion_for_normal_wind_tendency_approaching_cfl(
    levelmask: Field[[dims.KDim], bool],
    c_lin_e: Field[[dims.EdgeDim, E2CDim], wpfloat],
    z_w_con_c_full: fa.CellKField[vpfloat],
    ddqz_z_full_e: fa.EdgeKField[vpfloat],
    area_edge: fa.EdgeField[wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    zeta: fa.VertexKField[vpfloat],
    geofac_grdiv: Field[[dims.EdgeDim, E2C2EODim], wpfloat],
    vn: fa.EdgeKField[wpfloat],
    ddt_vn_apc: fa.EdgeKField[vpfloat],
    cfl_w_limit: vpfloat,
    scalfac_exdiff: wpfloat,
    dtime: wpfloat,
) -> fa.EdgeKField[vpfloat]:
    """Formerly known as _mo_velocity_advection_stencil_20."""
    z_w_con_c_full_wp, ddqz_z_full_e_wp, ddt_vn_apc_wp, cfl_w_limit_wp = astype(
        (z_w_con_c_full, ddqz_z_full_e, ddt_vn_apc, cfl_w_limit), wpfloat
    )

    w_con_e, difcoef = _init_two_edge_kdim_fields_with_zero_wp()

    w_con_e = where(
        levelmask | levelmask(Koff[1]),
        neighbor_sum(c_lin_e * z_w_con_c_full_wp(E2C), axis=E2CDim),
        w_con_e,
    )
    difcoef = where(
        (levelmask | levelmask(Koff[1]))
        & (abs(w_con_e) > astype(cfl_w_limit * ddqz_z_full_e, wpfloat)),
        scalfac_exdiff
        * minimum(
            wpfloat("0.85") - cfl_w_limit_wp * dtime,
            abs(w_con_e) * dtime / ddqz_z_full_e_wp - cfl_w_limit_wp * dtime,
        ),
        difcoef,
    )
    ddt_vn_apc_wp = where(
        (levelmask | levelmask(Koff[1]))
        & (abs(w_con_e) > astype(cfl_w_limit * ddqz_z_full_e, wpfloat)),
        ddt_vn_apc_wp
        + difcoef
        * area_edge
        * (
            neighbor_sum(geofac_grdiv * vn(E2C2EO), axis=E2C2EODim)
            + tangent_orientation
            * inv_primal_edge_length
            * astype(zeta(E2V[1]) - zeta(E2V[0]), wpfloat)
        ),
        ddt_vn_apc_wp,
    )
    return astype(ddt_vn_apc_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def add_extra_diffusion_for_normal_wind_tendency_approaching_cfl(
    levelmask: Field[[dims.KDim], bool],
    c_lin_e: Field[[dims.EdgeDim, E2CDim], wpfloat],
    z_w_con_c_full: fa.CellKField[vpfloat],
    ddqz_z_full_e: fa.EdgeKField[vpfloat],
    area_edge: fa.EdgeField[wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    zeta: fa.VertexKField[vpfloat],
    geofac_grdiv: Field[[dims.EdgeDim, E2C2EODim], wpfloat],
    vn: fa.EdgeKField[wpfloat],
    ddt_vn_apc: fa.EdgeKField[vpfloat],
    cfl_w_limit: vpfloat,
    scalfac_exdiff: wpfloat,
    dtime: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _add_extra_diffusion_for_normal_wind_tendency_approaching_cfl(
        levelmask,
        c_lin_e,
        z_w_con_c_full,
        ddqz_z_full_e,
        area_edge,
        tangent_orientation,
        inv_primal_edge_length,
        zeta,
        geofac_grdiv,
        vn,
        ddt_vn_apc,
        cfl_w_limit,
        scalfac_exdiff,
        dtime,
        out=ddt_vn_apc,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
