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
from gt4py.next.common import Field, GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import int32, maximum, where

from icon4py.model.atmosphere.dycore.mo_math_divrot_rot_vertex_ri_dsl import (
    _mo_math_divrot_rot_vertex_ri_dsl,
)
from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_19 import (
    _mo_velocity_advection_stencil_19,
)
from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_20 import (
    _mo_velocity_advection_stencil_20,
)
from icon4py.model.common.dimension import (
    CellDim,
    E2C2EODim,
    E2CDim,
    ECDim,
    EdgeDim,
    KDim,
    V2EDim,
    VertexDim,
)


@field_operator
def _fused_velocity_advection_stencil_19_to_20(
    vn: Field[[EdgeDim, KDim], float],
    geofac_rot: Field[[VertexDim, V2EDim], float],
    z_kin_hor_e: Field[[EdgeDim, KDim], float],
    coeff_gradekin: Field[[ECDim], float],
    z_ekinh: Field[[CellDim, KDim], float],
    vt: Field[[EdgeDim, KDim], float],
    f_e: Field[[EdgeDim], float],
    c_lin_e: Field[[EdgeDim, E2CDim], float],
    z_w_con_c_full: Field[[CellDim, KDim], float],
    vn_ie: Field[[EdgeDim, KDim], float],
    ddqz_z_full_e: Field[[EdgeDim, KDim], float],
    levelmask: Field[[KDim], bool],
    area_edge: Field[[EdgeDim], float],
    tangent_orientation: Field[[EdgeDim], float],
    inv_primal_edge_length: Field[[EdgeDim], float],
    geofac_grdiv: Field[[EdgeDim, E2C2EODim], float],
    k: Field[[KDim], int32],
    cfl_w_limit: float,
    scalfac_exdiff: float,
    d_time: float,
    extra_diffu: bool,
    nlev: int32,
    nrdmax: int32,
) -> Field[[EdgeDim, KDim], float]:
    zeta = _mo_math_divrot_rot_vertex_ri_dsl(vn, geofac_rot)

    ddt_vn_apc = _mo_velocity_advection_stencil_19(
        z_kin_hor_e,
        coeff_gradekin,
        z_ekinh,
        zeta,
        vt,
        f_e,
        c_lin_e,
        z_w_con_c_full,
        vn_ie,
        ddqz_z_full_e,
    )

    if extra_diffu:
      ddt_vn_apc = where( maximum(2, nrdmax - 2) <= k < nlev - 3,
              _mo_velocity_advection_stencil_20(
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
                  d_time,
              ),
              ddt_vn_apc,
          )

    return ddt_vn_apc


@program(grid_type=GridType.UNSTRUCTURED)
def fused_velocity_advection_stencil_19_to_20(
    vn: Field[[EdgeDim, KDim], float],
    geofac_rot: Field[[VertexDim, V2EDim], float],
    z_kin_hor_e: Field[[EdgeDim, KDim], float],
    coeff_gradekin: Field[[ECDim], float],
    z_ekinh: Field[[CellDim, KDim], float],
    vt: Field[[EdgeDim, KDim], float],
    f_e: Field[[EdgeDim], float],
    c_lin_e: Field[[EdgeDim, E2CDim], float],
    z_w_con_c_full: Field[[CellDim, KDim], float],
    vn_ie: Field[[EdgeDim, KDim], float],
    ddqz_z_full_e: Field[[EdgeDim, KDim], float],
    levelmask: Field[[KDim], bool],
    area_edge: Field[[EdgeDim], float],
    tangent_orientation: Field[[EdgeDim], float],
    inv_primal_edge_length: Field[[EdgeDim], float],
    geofac_grdiv: Field[[EdgeDim, E2C2EODim], float],
    ddt_vn_apc: Field[[EdgeDim, KDim], float],
    k: Field[[KDim], int32],
    cfl_w_limit: float,
    scalfac_exdiff: float,
    d_time: float,
    extra_diffu: bool,
    nlev: int32,
    nrdmax: int32,
):
    _fused_velocity_advection_stencil_19_to_20(
        vn,
        geofac_rot,
        z_kin_hor_e,
        coeff_gradekin,
        z_ekinh,
        vt,
        f_e,
        c_lin_e,
        z_w_con_c_full,
        vn_ie,
        ddqz_z_full_e,
        levelmask,
        area_edge,
        tangent_orientation,
        inv_primal_edge_length,
        geofac_grdiv,
        k,
        cfl_w_limit,
        scalfac_exdiff,
        d_time,
        extra_diffu,
        nlev,
        nrdmax,
        out=ddt_vn_apc,
    )
