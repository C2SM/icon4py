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
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.dycore.fused_velocity_advection_stencil_1_to_7 import (
    fused_velocity_advection_stencil_1_to_7,
)
from icon4py.model.common.dimension import CellDim, E2C2EDim, EdgeDim, KDim, V2CDim, VertexDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field

from .test_mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl_numpy,
)
from .test_mo_velocity_advection_stencil_01 import mo_velocity_advection_stencil_01_numpy
from .test_mo_velocity_advection_stencil_02 import mo_velocity_advection_stencil_02_numpy
from .test_mo_velocity_advection_stencil_03 import mo_velocity_advection_stencil_03_numpy
from .test_mo_velocity_advection_stencil_04 import mo_velocity_advection_stencil_04_numpy
from .test_mo_velocity_advection_stencil_05 import mo_velocity_advection_stencil_05_numpy
from .test_mo_velocity_advection_stencil_06 import mo_velocity_advection_stencil_06_numpy
from .test_mo_velocity_advection_stencil_07 import mo_velocity_advection_stencil_07_numpy


class TestFusedVelocityAdvectionStencil1To7(StencilTest):
    PROGRAM = fused_velocity_advection_stencil_1_to_7
    OUTPUTS = (
        "vt",
        "vn_ie",
        "z_kin_hor_e",
        "z_w_concorr_me",
        "z_v_grad_w",
    )

    @staticmethod
    def _fused_velocity_advection_stencil_1_to_6_numpy(
        mesh,
        vn,
        rbf_vec_coeff_e,
        wgtfac_e,
        ddxn_z_full,
        ddxt_z_full,
        z_w_concorr_me,
        wgtfacq_e_dsl,
        nflatlev,
        z_vt_ie,
        vt,
        vn_ie,
        z_kin_hor_e,
        k,
        nlevp1,
        lvn_only,
    ):

        k = k[np.newaxis, :]

        condition1 = k < nlevp1
        vt = np.where(
            condition1, mo_velocity_advection_stencil_01_numpy(mesh, vn, rbf_vec_coeff_e), vt
        )

        condition2 = (1 < k) & (k < nlevp1)
        vn_ie, z_kin_hor_e = np.where(
            condition2,
            mo_velocity_advection_stencil_02_numpy(mesh, wgtfac_e, vn, vt),
            (vn_ie, z_kin_hor_e),
        )

        if not lvn_only:
            z_vt_ie = np.where(
                condition2, mo_velocity_advection_stencil_03_numpy(mesh, wgtfac_e, vt), z_vt_ie
            )

        condition3 = k == 0
        vn_ie, z_vt_ie, z_kin_hor_e = np.where(
            condition3,
            mo_velocity_advection_stencil_05_numpy(vn, vt),
            (vn_ie, z_vt_ie, z_kin_hor_e),
        )

        condition4 = k == nlevp1
        vn_ie = np.where(
            condition4, mo_velocity_advection_stencil_06_numpy(wgtfacq_e_dsl, vn), vn_ie
        )

        condition5 = (nflatlev < k) & (k < nlevp1)
        z_w_concorr_me = np.where(
            condition5,
            mo_velocity_advection_stencil_04_numpy(vn, ddxn_z_full, ddxt_z_full, vt),
            z_w_concorr_me,
        )

        return vt, vn_ie, z_kin_hor_e, z_w_concorr_me

    @classmethod
    def reference(
        cls,
        mesh,
        vn,
        rbf_vec_coeff_e,
        wgtfac_e,
        ddxn_z_full,
        ddxt_z_full,
        z_w_concorr_me,
        wgtfacq_e_dsl,
        nflatlev,
        c_intp,
        w,
        inv_dual_edge_length,
        inv_primal_edge_length,
        tangent_orientation,
        z_vt_ie,
        vt,
        vn_ie,
        z_kin_hor_e,
        z_v_grad_w,
        k,
        istep,
        nlevp1,
        lvn_only,
        edge,
        lateral_boundary_7,
        halo_1,
        **kwargs,
    ):

        if istep == 1:
            (
                vt,
                vn_ie,
                z_kin_hor_e,
                z_w_concorr_me,
            ) = cls._fused_velocity_advection_stencil_1_to_6_numpy(
                mesh,
                vn,
                rbf_vec_coeff_e,
                wgtfac_e,
                ddxn_z_full,
                ddxt_z_full,
                z_w_concorr_me,
                wgtfacq_e_dsl,
                nflatlev,
                z_vt_ie,
                vt,
                vn_ie,
                z_kin_hor_e,
                k,
                nlevp1,
                lvn_only,
            )

        edge = edge[:, np.newaxis]

        z_w_v = mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl_numpy(mesh, w, c_intp)

        condition_mask = (lateral_boundary_7 < edge) & (edge < halo_1) & (k < nlevp1)

        if not lvn_only:
            z_v_grad_w = np.where(
                condition_mask,
                mo_velocity_advection_stencil_07_numpy(
                    mesh,
                    vn_ie,
                    inv_dual_edge_length,
                    w,
                    z_vt_ie,
                    inv_primal_edge_length,
                    tangent_orientation,
                    z_w_v,
                ),
                z_v_grad_w,
            )

        return dict(
            vt=vt,
            vn_ie=vn_ie,
            z_kin_hor_e=z_kin_hor_e,
            z_w_concorr_me=z_w_concorr_me,
            z_v_grad_w=z_v_grad_w,
        )

    @pytest.fixture
    def input_data(self, mesh):
        c_intp = random_field(mesh, VertexDim, V2CDim)
        vn = random_field(mesh, EdgeDim, KDim)
        rbf_vec_coeff_e = random_field(mesh, EdgeDim, E2C2EDim)
        vt = zero_field(mesh, EdgeDim, KDim)
        wgtfac_e = random_field(mesh, EdgeDim, KDim)
        vn_ie = zero_field(mesh, EdgeDim, KDim)
        z_kin_hor_e = zero_field(mesh, EdgeDim, KDim)
        z_vt_ie = zero_field(mesh, EdgeDim, KDim)
        ddxn_z_full = random_field(mesh, EdgeDim, KDim)
        ddxt_z_full = random_field(mesh, EdgeDim, KDim)
        z_w_concorr_me = zero_field(mesh, EdgeDim, KDim)
        inv_dual_edge_length = random_field(mesh, EdgeDim)
        w = random_field(mesh, CellDim, KDim)
        inv_primal_edge_length = random_field(mesh, EdgeDim)
        tangent_orientation = random_field(mesh, EdgeDim)
        z_v_grad_w = zero_field(mesh, EdgeDim, KDim)
        wgtfacq_e = random_field(mesh, EdgeDim, KDim)

        k = zero_field(mesh, KDim, dtype=int32)
        for level in range(mesh.k_level):
            k[level] = level

        edge = zero_field(mesh, EdgeDim, dtype=int32)
        for e in range(mesh.n_edges):
            edge[e] = e

        nlevp1 = mesh.k_level + 1
        nflatlev = 13

        istep = 1
        lvn_only = False

        lateral_boundary_7 = 2
        halo_1 = 6

        return dict(
            vn=vn,
            rbf_vec_coeff_e=rbf_vec_coeff_e,
            wgtfac_e=wgtfac_e,
            ddxn_z_full=ddxn_z_full,
            ddxt_z_full=ddxt_z_full,
            z_w_concorr_me=z_w_concorr_me,
            wgtfacq_e_dsl=wgtfacq_e,
            nflatlev=nflatlev,
            c_intp=c_intp,
            w=w,
            inv_dual_edge_length=inv_dual_edge_length,
            inv_primal_edge_length=inv_primal_edge_length,
            tangent_orientation=tangent_orientation,
            z_vt_ie=z_vt_ie,
            vt=vt,
            vn_ie=vn_ie,
            z_kin_hor_e=z_kin_hor_e,
            z_v_grad_w=z_v_grad_w,
            k=k,
            istep=istep,
            nlevp1=nlevp1,
            lvn_only=lvn_only,
            edge=edge,
            lateral_boundary_7=lateral_boundary_7,
            halo_1=halo_1,
        )
