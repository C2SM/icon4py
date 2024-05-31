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
from gt4py.next.ffront.fbuiltins import (
    Field,
    abs,
    astype,
    int32,
    minimum,
    neighbor_sum,
    where,
)
from model.common.tests import field_type_aliases as fa

from icon4py.model.atmosphere.dycore.init_two_edge_kdim_fields_with_zero_wp import (
    _init_two_edge_kdim_fields_with_zero_wp,
)
from icon4py.model.common.dimension import (
    E2C,
    E2C2EO,
    E2V,
    E2C2EODim,
    E2CDim,
    EdgeDim,
    KDim,
    Koff,
)
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _add_extra_diffusion_for_normal_wind_tendency_approaching_cfl(
    levelmask: Field[[KDim], bool],
    c_lin_e: Field[[EdgeDim, E2CDim], wpfloat],
    z_w_con_c_full: fa.CKvpField,
    ddqz_z_full_e: Field[[EdgeDim, KDim], vpfloat],
    area_edge: fa.EwpField,
    tangent_orientation: fa.EwpField,
    inv_primal_edge_length: fa.EwpField,
    zeta: fa.VKvpField,
    geofac_grdiv: Field[[EdgeDim, E2C2EODim], wpfloat],
    vn: fa.EKwpField,
    ddt_vn_apc: Field[[EdgeDim, KDim], vpfloat],
    cfl_w_limit: vpfloat,
    scalfac_exdiff: wpfloat,
    dtime: wpfloat,
) -> Field[[EdgeDim, KDim], vpfloat]:
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
    levelmask: Field[[KDim], bool],
    c_lin_e: Field[[EdgeDim, E2CDim], wpfloat],
    z_w_con_c_full: fa.CKvpField,
    ddqz_z_full_e: Field[[EdgeDim, KDim], vpfloat],
    area_edge: fa.EwpField,
    tangent_orientation: fa.EwpField,
    inv_primal_edge_length: fa.EwpField,
    zeta: fa.VKvpField,
    geofac_grdiv: Field[[EdgeDim, E2C2EODim], wpfloat],
    vn: fa.EKwpField,
    ddt_vn_apc: Field[[EdgeDim, KDim], vpfloat],
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
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
