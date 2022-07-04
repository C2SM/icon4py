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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_30 import (
    mo_solve_nonhydro_stencil_30_vt,
    mo_solve_nonhydro_stencil_30_z_graddiv_vn,
    mo_solve_nonhydro_stencil_30_z_vn_avg,
)
from icon4py.common.dimension import E2C2EDim, E2C2EODim, EdgeDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, zero_field


def mo_solve_nonhydro_stencil_30_z_vn_avg_numpy(
    e2c2eO: np.array, e_flx_avg: np.array, vn: np.array
) -> np.array:
    geofac_grdiv = np.expand_dims(e_flx_avg, axis=-1)
    z_vn_avg = np.sum(vn[e2c2eO] * geofac_grdiv, axis=1)
    return z_vn_avg


def test_mo_solve_nonhydro_stencil_30_z_vn_avg():
    mesh = SimpleMesh()

    e_flx_avg = random_field(mesh, EdgeDim, E2C2EODim)
    vn = random_field(mesh, EdgeDim, KDim)
    z_vn_avg = zero_field(mesh, EdgeDim, KDim)

    ref = mo_solve_nonhydro_stencil_30_z_vn_avg_numpy(
        mesh.e2c2eO, np.asarray(e_flx_avg), np.asarray(vn)
    )
    mo_solve_nonhydro_stencil_30_z_vn_avg(
        e_flx_avg,
        vn,
        z_vn_avg,
        offset_provider={"E2C2EO": mesh.get_e2c2eO_offset_provider()},
    )
    assert np.allclose(z_vn_avg, ref)


def mo_solve_nonhydro_stencil_30_z_graddiv_vn_numpy(
    e2c2eO: np.array, geofac_grdiv: np.array, vn: np.array
) -> np.array:
    geofac_grdiv = np.expand_dims(geofac_grdiv, axis=-1)
    z_graddiv_vn = np.sum(vn[e2c2eO] * geofac_grdiv, axis=1)
    return z_graddiv_vn


def test_mo_solve_nonhydro_stencil_30_z_graddiv_vng():
    mesh = SimpleMesh()

    geofac_grdiv = random_field(mesh, EdgeDim, E2C2EODim)
    vn = random_field(mesh, EdgeDim, KDim)
    z_graddiv_vn = zero_field(mesh, EdgeDim, KDim)

    ref = mo_solve_nonhydro_stencil_30_z_graddiv_vn_numpy(
        mesh.e2c2eO, np.asarray(geofac_grdiv), np.asarray(vn)
    )
    mo_solve_nonhydro_stencil_30_z_graddiv_vn(
        geofac_grdiv,
        vn,
        z_graddiv_vn,
        offset_provider={"E2C2EO": mesh.get_e2c2eO_offset_provider()},
    )
    assert np.allclose(z_graddiv_vn, ref)


def mo_solve_nonhydro_stencil_30_vt_numpy(
    e2c2e: np.array, rbf_vec_coeff_e: np.array, vn: np.array
) -> np.array:
    rbf_vec_coeff_e = np.expand_dims(rbf_vec_coeff_e, axis=-1)
    vt = np.sum(vn[e2c2e] * rbf_vec_coeff_e, axis=1)
    return vt


def test_mo_solve_nonhydro_stencil_30_vt():
    mesh = SimpleMesh()

    rbf_vec_coeff_e = random_field(mesh, EdgeDim, E2C2EDim)
    vn = random_field(mesh, EdgeDim, KDim)
    vt = zero_field(mesh, EdgeDim, KDim)

    ref = mo_solve_nonhydro_stencil_30_vt_numpy(
        mesh.e2c2e, np.asarray(rbf_vec_coeff_e), np.asarray(vn)
    )
    mo_solve_nonhydro_stencil_30_vt(
        rbf_vec_coeff_e,
        vn,
        vt,
        offset_provider={"E2C2E": mesh.get_e2c2e_offset_provider()},
    )
    assert np.allclose(vt, ref)
