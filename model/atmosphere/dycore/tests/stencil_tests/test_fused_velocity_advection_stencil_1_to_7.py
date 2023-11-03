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

from icon4py.model.atmosphere.dycore.fused_velocity_advection_stencil_1_to_7 import (fused_velocity_advection_stencil_1_to_7)
from icon4py.model.common.dimension import CellDim, EdgeDim, VertexDim, E2C2EDim, V2CDim, KDim

from icon4py.model.common.test_utils.helpers import random_field, zero_field, StencilTest


class TestFusedVelocityAdvectionStencil1To7(StencilTest):
    PROGRAM = fused_velocity_advection_stencil_1_to_7
    OUTPUTS = ("vt", "vn_ie", "z_kin_hor_e", "z_w_concorr_me", "z_v_grad_w",)

    @staticmethod
    def reference(
        mesh,
        **kwargs,
    ) -> tuple[np.array]:

        vt = 0.
        vn_ie = 0.
        z_kin_hor_e = 0.
        z_w_concorr_me = 0.
        z_v_grad_w = 0.

        return dict(vt=vt, vn_ie=vn_ie, z_kin_hor_e=z_kin_hor_e, z_w_concorr_me=z_w_concorr_me, z_v_grad_w=z_v_grad_w)

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

        vert_idx = zero_field(mesh, KDim, dtype=int32)
        for level in range(mesh.k_level):
            vert_idx[level] = level

        horz_idx = zero_field(mesh, EdgeDim, dtype=int32)
        for edge in range(mesh.n_edges):
            horz_idx[edge] = edge

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
            vert_idx=vert_idx,
            istep=istep,
            nlevp1=nlevp1,
            lvn_only=lvn_only,
            horz_idx=horz_idx,
            lateral_boundary_7=lateral_boundary_7,
            halo_1=halo_1,
        )
