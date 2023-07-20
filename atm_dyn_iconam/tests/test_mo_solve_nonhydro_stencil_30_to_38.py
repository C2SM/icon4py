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
import pytest

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_30_to_38 import (
    mo_solve_nonhydro_stencil_30_to_38,
)
from icon4py.common.dimension import E2C2EDim, E2C2EODim, EdgeDim, KDim

from . import (
    test_mo_solve_nonhydro_stencil_30,
    test_mo_solve_nonhydro_stencil_31,
    test_mo_solve_nonhydro_stencil_32,
    test_mo_solve_nonhydro_stencil_34,
)
from .test_utils.helpers import random_field, zero_field
from .test_utils.stencil_test import StencilTest


@pytest.fixture(ids=["istep=0", "istep=1"], params=[0, 1])
def istep(request):
    return request.param


@pytest.fixture(ids=["lclean_mflx=True", "lclean_mflx=False"], params=[True, False])
def lclean_mflx(request):
    return request.param


class TestMoSolveNonhydroStencil30_to_38(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_30_to_38
    OUTPUTS = (
        "z_vn_avg",
        "z_graddiv_vn",
        "vt",
        "mass_fl_e",
        "z_theta_v_fl_e",
        "vn_traj",
        "mass_flx_me",
    )

    @staticmethod
    def reference(
        mesh,
        istep: int,
        lclean_mflx: bool,
        e_flx_avg: np.array,
        vn: np.array,
        geofac_grdiv: np.array,
        rbf_vec_coeff_e: np.array,
        z_rho_e: np.array,
        z_vn_avg: np.array,
        ddqz_z_full_e: np.array,
        z_theta_v_e: np.array,
        vt: np.array,
        z_graddiv_vn: np.array,
        mass_fl_e: np.array,
        vn_traj: np.array,
        mass_flx_me: np.array,
        r_nsubsteps: float,
        **kwargs,
    ) -> dict:
        if istep == 0:
            (
                z_vn_avg,
                z_graddiv_vn,
                vt,
            ) = test_mo_solve_nonhydro_stencil_30.TestMoSolveNonhydroStencil30.reference(
                mesh, e_flx_avg, vn, geofac_grdiv, rbf_vec_coeff_e
            ).values()
        else:
            z_vn_avg = test_mo_solve_nonhydro_stencil_31.TestMoSolveNonhydroStencil31.reference(
                mesh, e_flx_avg, vn
            )[
                "z_vn_avg"
            ]
        (
            mass_fl_e,
            z_theta_v_fl_e,
        ) = test_mo_solve_nonhydro_stencil_32.TestMoSolveNonhydroStencil32.reference(
            mesh, z_rho_e, z_vn_avg, ddqz_z_full_e, z_theta_v_e
        ).values()

        if lclean_mflx:
            vn_traj = np.zeros_like(vn_traj)
            mass_flx_me = np.zeros_like(mass_flx_me)

        (
            vn_traj,
            mass_flx_me,
        ) = test_mo_solve_nonhydro_stencil_34.TestMoSolveNonhydroStencil34.reference(
            mesh, z_vn_avg, mass_fl_e, vn_traj, mass_flx_me, r_nsubsteps
        ).values()

        return dict(
            z_vn_avg=z_vn_avg,
            z_graddiv_vn=z_graddiv_vn,
            vt=vt,
            mass_fl_e=mass_fl_e,
            z_theta_v_fl_e=z_theta_v_fl_e,
            vn_traj=vn_traj,
            mass_flx_me=mass_flx_me,
        )

    @pytest.fixture
    def input_data(self, mesh, istep, lclean_mflx):
        e_flx_avg = random_field(mesh, EdgeDim, E2C2EODim)
        geofac_grdiv = random_field(mesh, EdgeDim, E2C2EODim)
        rbf_vec_coeff_e = random_field(mesh, EdgeDim, E2C2EDim)
        vn = random_field(mesh, EdgeDim, KDim)
        z_rho_e = random_field(mesh, EdgeDim, KDim)
        ddqz_z_full_e = random_field(mesh, EdgeDim, KDim)
        z_theta_v_e = random_field(mesh, EdgeDim, KDim)
        mass_flx_me = random_field(mesh, EdgeDim, KDim)
        vn_traj = random_field(mesh, EdgeDim, KDim)
        z_vn_avg = zero_field(mesh, EdgeDim, KDim)
        z_graddiv_vn = zero_field(mesh, EdgeDim, KDim)
        vt = zero_field(mesh, EdgeDim, KDim)
        mass_fl_e = zero_field(mesh, EdgeDim, KDim)
        z_theta_v_fl_e = zero_field(mesh, EdgeDim, KDim)
        r_nsubsteps = 9.0

        return dict(
            istep=istep,
            lclean_mflx=lclean_mflx,
            e_flx_avg=e_flx_avg,
            vn=vn,
            geofac_grdiv=geofac_grdiv,
            rbf_vec_coeff_e=rbf_vec_coeff_e,
            z_rho_e=z_rho_e,
            ddqz_z_full_e=ddqz_z_full_e,
            z_theta_v_e=z_theta_v_e,
            mass_flx_me=mass_flx_me,
            vn_traj=vn_traj,
            z_vn_avg=z_vn_avg,
            z_graddiv_vn=z_graddiv_vn,
            vt=vt,
            mass_fl_e=mass_fl_e,
            z_theta_v_fl_e=z_theta_v_fl_e,
            r_nsubsteps=r_nsubsteps,
        )
