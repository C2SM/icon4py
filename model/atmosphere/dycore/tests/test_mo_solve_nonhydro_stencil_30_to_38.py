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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_30_to_38 import (
    mo_solve_nonhydro_stencil_30_to_38,
    mo_solve_nonhydro_stencil_36_to_38,
)
from icon4py.model.common.dimension import E2C2EDim, E2C2EODim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    index_field,
    random_field,
    zero_field,
)

from . import (
    test_mo_solve_nonhydro_stencil_30,
    test_mo_solve_nonhydro_stencil_31,
    test_mo_solve_nonhydro_stencil_32,
    test_mo_solve_nonhydro_stencil_34,
    test_mo_solve_nonhydro_stencil_35,
)


@pytest.fixture(ids=["istep=0", "istep=1"], params=[0, 1])
def istep(request):
    return request.param


@pytest.fixture(ids=["lclean_mflx=True", "lclean_mflx=False"], params=[True, False])
def lclean_mflx(request):
    return request.param


class TestMoSolveNonhydroStencil36_to_38(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_36_to_38
    OUTPUTS = ("vn_ie", "z_vt_ie", "z_kin_hor_e")

    @staticmethod
    def reference(
        mesh,
        wgtfac_e: np.array,
        vn: np.array,
        vt: np.array,
        vn_ie: np.array,
        z_vt_ie: np.array,
        z_kin_hor_e: np.array,
        **kwargs,
    ):
        vn_ie[:, 0] = vn[:, 0]
        z_vt_ie[:, 0] = vt[:, 0]
        z_kin_hor_e[:, 0] = 0.5 * (pow(vn[:, 0], 2) + pow(vt[:, 0], 2))

        vn_offset_1 = np.roll(vn, shift=1, axis=1)
        vt_offset_1 = np.roll(vt, shift=1, axis=1)

        vn_ie[:, 1:-1] = (wgtfac_e * vn + (1.0 - wgtfac_e) * vn_offset_1)[:, 1:-1]
        z_vt_ie[:, 1:-1] = (wgtfac_e * vt + (1.0 - wgtfac_e) * vt_offset_1)[:, 1:-1]
        z_kin_hor_e[:, 1:-1] = (0.5 * (vn**2 + vt**2))[:, 1:-1]

        vn_ie[:, -1] = (
            wgtfac_e[:, -2] * vn[:, -2] + wgtfac_e[:, -3] * vn[:, -3] + wgtfac_e[:, -4] * vn[:, -4]
        )
        # z_vt_ie, z_kin_hor_e should probably be undefined (here they are implicit still 0.)

        return dict(vn_ie=vn_ie, z_vt_ie=z_vt_ie, z_kin_hor_e=z_kin_hor_e)

    @pytest.fixture
    def input_data(self, mesh):
        wgtfac_e = random_field(mesh, EdgeDim, KDim)
        vn = random_field(mesh, EdgeDim, KDim)
        vt = random_field(mesh, EdgeDim, KDim)

        vn_ie = zero_field(mesh, EdgeDim, KDim)
        z_vt_ie = zero_field(mesh, EdgeDim, KDim)
        z_kin_hor_e = zero_field(mesh, EdgeDim, KDim)
        return dict(
            wgtfac_e=wgtfac_e,
            vn=vn,
            vt=vt,
            vn_ie=vn_ie,
            z_vt_ie=z_vt_ie,
            z_kin_hor_e=z_kin_hor_e,
            k=index_field(mesh, KDim),
            nlev=mesh.k_level,
        )


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
        "z_w_concorr_me",
        "vn_ie",
        "z_vt_ie",
        "z_kin_hor_e",
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
        ddxn_z_full: np.array,
        ddxt_z_full: np.array,
        mass_flx_me: np.array,
        wgtfac_e: np.array,
        vn_ie: np.array,
        z_vt_ie: np.array,
        z_kin_hor_e: np.array,
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
            )["z_vn_avg"]
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

        z_w_concorr_me = test_mo_solve_nonhydro_stencil_35.TestMoSolveNonhydroStencil35.reference(
            mesh, vn, ddxn_z_full, ddxt_z_full, vt
        )["z_w_concorr_me"]

        (vn_ie, z_vt_ie, z_kin_hor_e,) = TestMoSolveNonhydroStencil36_to_38.reference(
            mesh, wgtfac_e, vn, vt, vn_ie, z_vt_ie, z_kin_hor_e
        ).values()

        return dict(
            z_vn_avg=z_vn_avg,
            z_graddiv_vn=z_graddiv_vn,
            vt=vt,
            mass_fl_e=mass_fl_e,
            z_theta_v_fl_e=z_theta_v_fl_e,
            vn_traj=vn_traj,
            mass_flx_me=mass_flx_me,
            z_w_concorr_me=z_w_concorr_me,
            vn_ie=vn_ie,
            z_vt_ie=z_vt_ie,
            z_kin_hor_e=z_kin_hor_e,
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
        ddxn_z_full = random_field(mesh, EdgeDim, KDim)
        ddxt_z_full = random_field(mesh, EdgeDim, KDim)
        wgtfac_e = random_field(mesh, EdgeDim, KDim)
        z_vn_avg = zero_field(mesh, EdgeDim, KDim)
        z_graddiv_vn = zero_field(mesh, EdgeDim, KDim)
        vt = zero_field(mesh, EdgeDim, KDim)
        mass_fl_e = zero_field(mesh, EdgeDim, KDim)
        z_theta_v_fl_e = zero_field(mesh, EdgeDim, KDim)
        z_w_concorr_me = zero_field(mesh, EdgeDim, KDim)
        vn_ie = zero_field(mesh, EdgeDim, KDim)
        z_vt_ie = zero_field(mesh, EdgeDim, KDim)
        z_kin_hor_e = zero_field(mesh, EdgeDim, KDim)
        r_nsubsteps = 9.0
        k = index_field(mesh, KDim)
        nlev = mesh.k_level

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
            ddxn_z_full=ddxn_z_full,
            ddxt_z_full=ddxt_z_full,
            wgtfac_e=wgtfac_e,
            z_vn_avg=z_vn_avg,
            z_graddiv_vn=z_graddiv_vn,
            vt=vt,
            mass_fl_e=mass_fl_e,
            z_theta_v_fl_e=z_theta_v_fl_e,
            z_w_concorr_me=z_w_concorr_me,
            vn_ie=vn_ie,
            z_vt_ie=z_vt_ie,
            z_kin_hor_e=z_kin_hor_e,
            r_nsubsteps=r_nsubsteps,
            k=k,
            nlev=nlev,
        )
