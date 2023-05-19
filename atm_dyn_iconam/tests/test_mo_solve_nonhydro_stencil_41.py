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
from simple_mesh import SimpleMesh
from utils import random_field, zero_field

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_41 import (
    mo_solve_nonhydro_stencil_41,
)
from icon4py.common.dimension import C2EDim, CellDim, EdgeDim, KDim


def mo_solve_nonhydro_stencil_41_numpy(
    c2e: np.array, geofac_div: np.array, mass_fl_e: np.array, z_theta_v_fl_e: np.array
) -> tuple[np.array]:
    geofac_div = np.expand_dims(geofac_div, axis=-1)
    z_flxdiv_mass = np.sum(geofac_div * mass_fl_e[c2e], axis=1)
    z_flxdiv_theta = np.sum(geofac_div * z_theta_v_fl_e[c2e], axis=1)
    return z_flxdiv_mass, z_flxdiv_theta


def test_mo_solve_nonhydro_stencil_41_z_flxdiv_theta():
    mesh = SimpleMesh()

    geofac_div = random_field(mesh, CellDim, C2EDim)
    z_theta_v_fl_e = random_field(mesh, EdgeDim, KDim)
    z_flxdiv_theta = zero_field(mesh, CellDim, KDim)
    mass_fl_e = random_field(mesh, EdgeDim, KDim)
    z_flxdiv_mass = zero_field(mesh, CellDim, KDim)

    z_flxdiv_mass_ref, z_flxdiv_theta_ref = mo_solve_nonhydro_stencil_41_numpy(
        mesh.c2e,
        np.asarray(geofac_div),
        np.asarray(mass_fl_e),
        np.asarray(z_theta_v_fl_e),
    )
    mo_solve_nonhydro_stencil_41(
        geofac_div,
        mass_fl_e,
        z_theta_v_fl_e,
        z_flxdiv_mass,
        z_flxdiv_theta,
        offset_provider={"C2E": mesh.get_c2e_offset_provider()},
    )

    assert np.allclose(z_flxdiv_mass, z_flxdiv_mass_ref)
    assert np.allclose(z_flxdiv_theta, z_flxdiv_theta_ref)
