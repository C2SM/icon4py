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

import numpy as np

from icon4py.model.atm_dyn_iconam.mo_velocity_advection_stencil_20 import (
    mo_velocity_advection_stencil_20,
)
from icon4py.model.common.dimension import (
    CellDim,
    E2C2EODim,
    E2CDim,
    EdgeDim,
    KDim,
    VertexDim,
)

from icon4py.model.common.test_utils.helpers import random_field, random_mask
from icon4py.model.common.test_utils.simple_mesh import SimpleMesh


def mo_velocity_advection_stencil_20_numpy(
    e2c: np.array,
    e2c2eO: np.array,
    e2v: np.array,
    levelmask: np.array,
    c_lin_e: np.array,
    z_w_con_c_full: np.array,
    ddqz_z_full_e: np.array,
    area_edge: np.array,
    tangent_orientation: np.array,
    inv_primal_edge_length: np.array,
    zeta: np.array,
    geofac_grdiv: np.array,
    vn: np.array,
    ddt_vn_adv: np.array,
    cfl_w_limit,
    scalfac_exdiff,
    d_time,
):
    w_con_e = np.zeros_like(vn)
    difcoef = np.zeros_like(vn)

    levelmask_offset_1 = np.roll(levelmask, shift=-1, axis=0)

    c_lin_e = np.expand_dims(c_lin_e, axis=-1)
    geofac_grdiv = np.expand_dims(geofac_grdiv, axis=-1)
    area_edge = np.expand_dims(area_edge, axis=-1)
    tangent_orientation = np.expand_dims(tangent_orientation, axis=-1)
    inv_primal_edge_length = np.expand_dims(inv_primal_edge_length, axis=-1)

    w_con_e = np.where(
        (levelmask == 1) | (levelmask_offset_1 == 1),
        np.sum(c_lin_e * z_w_con_c_full[e2c], axis=1),
        w_con_e,
    )
    difcoef = np.where(
        ((levelmask == 1) | (levelmask_offset_1 == 1))
        & (np.abs(w_con_e) > cfl_w_limit * ddqz_z_full_e),
        scalfac_exdiff
        * np.minimum(
            0.85 - cfl_w_limit * d_time,
            np.abs(w_con_e) * d_time / ddqz_z_full_e - cfl_w_limit * d_time,
        ),
        difcoef,
    )
    ddt_vn_adv = np.where(
        ((levelmask == 1) | (levelmask_offset_1 == 1))
        & (np.abs(w_con_e) > cfl_w_limit * ddqz_z_full_e),
        ddt_vn_adv
        + difcoef * area_edge * np.sum(geofac_grdiv * vn[e2c2eO], axis=1)
        + tangent_orientation * inv_primal_edge_length * np.sum(zeta[e2v], axis=1),
        ddt_vn_adv,
    )
    return ddt_vn_adv


def test_mo_velocity_advection_stencil_20():
    mesh = SimpleMesh()

    levelmask = random_mask(mesh, KDim)
    c_lin_e = random_field(mesh, EdgeDim, E2CDim)
    z_w_con_c_full = random_field(mesh, CellDim, KDim)
    ddqz_z_full_e = random_field(mesh, EdgeDim, KDim)
    area_edge = random_field(mesh, EdgeDim)
    tangent_orientation = random_field(mesh, EdgeDim)
    inv_primal_edge_length = random_field(mesh, EdgeDim)
    zeta = random_field(mesh, VertexDim, KDim)
    geofac_grdiv = random_field(mesh, EdgeDim, E2C2EODim)
    vn = random_field(mesh, EdgeDim, KDim)
    ddt_vn_adv = random_field(mesh, EdgeDim, KDim)
    cfl_w_limit = 4.0
    scalfac_exdiff = 6.0
    d_time = 2.0

    ddt_vn_adv_ref = mo_velocity_advection_stencil_20_numpy(
        mesh.e2c,
        mesh.e2c2eO,
        mesh.e2v,
        np.asarray(levelmask),
        np.asarray(c_lin_e),
        np.asarray(z_w_con_c_full),
        np.asarray(ddqz_z_full_e),
        np.asarray(area_edge),
        np.asarray(tangent_orientation),
        np.asarray(inv_primal_edge_length),
        np.asarray(zeta),
        np.asarray(geofac_grdiv),
        np.asarray(vn),
        np.asarray(ddt_vn_adv),
        cfl_w_limit,
        scalfac_exdiff,
        d_time,
    )

    mo_velocity_advection_stencil_20(
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
        ddt_vn_adv,
        cfl_w_limit,
        scalfac_exdiff,
        d_time,
        offset_provider={
            "Koff": KDim,
            "E2C": mesh.get_e2c_offset_provider(),
            "E2C2EO": mesh.get_e2c2eO_offset_provider(),
            "E2V": mesh.get_e2v_offset_provider(),
        },
    )

    assert np.allclose(ddt_vn_adv[:, :-1], ddt_vn_adv_ref[:, :-1])
