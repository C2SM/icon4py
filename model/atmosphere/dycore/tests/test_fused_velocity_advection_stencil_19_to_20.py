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

from icon4py.model.atmosphere.dycore.fused_velocity_advection_stencil_19_to_20 import (fused_velocity_advection_stencil_19_to_20)
from icon4py.model.common.dimension import CellDim, EdgeDim, VertexDim, E2CDim, ECDim, E2C2EODim, V2EDim, KDim

from icon4py.model.common.test_utils.helpers import as_1D_sparse_field, random_field, random_mask, zero_field, StencilTest


class TestFusedVelocityAdvectionStencil19To20(StencilTest):
    PROGRAM = fused_velocity_advection_stencil_19_to_20
    OUTPUTS = ("ddt_vn_adv",)

    @staticmethod
    def reference(
        mesh,
        **kwargs,
    ) -> tuple[np.array]:
        ddt_vn_adv = 0.
        return dict(ddt_vn_adv=ddt_vn_adv)

    @pytest.fixture
    def input_data(self, mesh):
        z_kin_hor_e = random_field(mesh, EdgeDim, KDim)
        coeff_gradekin = random_field(mesh, EdgeDim, E2CDim)
        coeff_gradekin_new = as_1D_sparse_field(coeff_gradekin, ECDim)
        z_ekinh = random_field(mesh, CellDim, KDim)
        vt = random_field(mesh, EdgeDim, KDim)
        f_e = random_field(mesh, EdgeDim)
        c_lin_e = random_field(mesh, EdgeDim, E2CDim)
        z_w_con_c_full = random_field(mesh, CellDim, KDim)
        vn_ie = random_field(mesh, EdgeDim, KDim, extend={KDim: 1})
        ddqz_z_full_e = random_field(mesh, EdgeDim, KDim)
        ddt_vn_adv = zero_field(mesh, EdgeDim, KDim)
        levelmask = random_mask(mesh, KDim, extend={KDim: 1})
        area_edge = random_field(mesh, EdgeDim)
        tangent_orientation = random_field(mesh, EdgeDim)
        inv_primal_edge_length = random_field(mesh, EdgeDim)
        geofac_grdiv = random_field(mesh, EdgeDim, E2C2EODim)
        vn = random_field(mesh, EdgeDim, KDim)
        geofac_rot = random_field(mesh, VertexDim, V2EDim)
        cfl_w_limit = 4.0
        scalfac_exdiff = 6.0
        d_time = 2.0

        vert_idx = zero_field(mesh, KDim, dtype=int32)
        for level in range(mesh.k_level):
            vert_idx[level] = level

        nlev = mesh.k_level
        nrdmax = 5
        extra_diffu = True

        return dict(
            vn=vn,
            geofac_rot=geofac_rot,
            z_kin_hor_e=z_kin_hor_e,
            coeff_gradekin=coeff_gradekin_new,
            z_ekinh=z_ekinh,
            vt=vt,
            f_e=f_e,
            c_lin_e=c_lin_e,
            z_w_con_c_full=z_w_con_c_full,
            vn_ie=vn_ie,
            ddqz_z_full_e=ddqz_z_full_e,
            levelmask=levelmask,
            area_edge=area_edge,
            tangent_orientation=tangent_orientation,
            inv_primal_edge_length=inv_primal_edge_length,
            geofac_grdiv=geofac_grdiv,
            vert_idx=vert_idx,
            cfl_w_limit=cfl_w_limit,
            scalfac_exdiff=scalfac_exdiff,
            d_time=d_time,
            extra_diffu=extra_diffu,
            nlev=nlev,
            nrdmax=nrdmax,
            ddt_vn_adv=ddt_vn_adv,
        )
