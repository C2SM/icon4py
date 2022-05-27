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

from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_05 import (
    mo_velocity_advection_stencil_05_vn_ie,
    mo_velocity_advection_stencil_05_z_kin_hor_e,
    mo_velocity_advection_stencil_05_z_vt_ie,
)
from icon4py.common.dimension import C2EDim, CellDim, EdgeDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, zero_field


def mo_velocity_advection_stencil_05_numpy_vn_ie(vn: np.array):
    vn_ie = vn
    return vn_ie


def mo_velocity_advection_stencil_05_numpy_z_vt_ie(vt: np.array):
    z_vt_ie = vt
    return z_vt_ie


def mo_velocity_advection_stencil_05_numpy_z_kin_hor_e(vn: np.array, vt: np.array):
    z_kin_hor_e = 0.5 * ((vn * vn) + (vt * vt))
    return z_kin_hor_e


def test_mo_velocity_advection_stencil_05_vn_ie():
    mesh = SimpleMesh()

    vn = random_field(mesh, EdgeDim, KDim)
    vn_ie = zero_field(mesh, EdgeDim, KDim)

    ref = mo_velocity_advection_stencil_05_numpy_vn_ie(np.asarray(vn))
    mo_velocity_advection_stencil_05_vn_ie(
        vn,
        vn_ie,
        offset_provider={},
    )
    assert np.allclose(vn_ie, ref)


def test_mo_velocity_advection_stencil_05_z_vt_ie():
    mesh = SimpleMesh()

    vt = random_field(mesh, EdgeDim, KDim)
    z_vt_ie = zero_field(mesh, EdgeDim, KDim)

    ref = mo_velocity_advection_stencil_05_numpy_z_vt_ie(np.asarray(vt))
    mo_velocity_advection_stencil_05_z_vt_ie(
        vt,
        z_vt_ie,
        offset_provider={},
    )
    assert np.allclose(z_vt_ie, ref)


def test_mo_velocity_advection_stencil_05_z_kin_hor_e():
    mesh = SimpleMesh()

    vn = random_field(mesh, EdgeDim, KDim)
    vt = random_field(mesh, EdgeDim, KDim)
    z_kin_hor_e = zero_field(mesh, EdgeDim, KDim)

    ref = mo_velocity_advection_stencil_05_numpy_z_kin_hor_e(
        np.asarray(vn), np.asarray(vt)
    )
    mo_velocity_advection_stencil_05_z_kin_hor_e(
        vn,
        vt,
        z_kin_hor_e,
        offset_provider={},
    )
    assert np.allclose(z_kin_hor_e, ref)
