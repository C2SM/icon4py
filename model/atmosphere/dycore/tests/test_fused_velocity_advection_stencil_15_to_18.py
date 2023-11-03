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

from icon4py.model.atmosphere.dycore.fused_velocity_advection_stencil_15_to_18 import (fused_velocity_advection_stencil_15_to_18)
from icon4py.model.common.dimension import CellDim, EdgeDim, C2EDim, C2E2CODim, KDim

from icon4py.model.common.test_utils.helpers import random_field, random_mask, zero_field, StencilTest


class TestFusedVelocityAdvectionStencil19To20(StencilTest):
    PROGRAM = fused_velocity_advection_stencil_15_to_18
    OUTPUTS = ("z_w_con_c_full", "ddt_w_adv",)

    @staticmethod
    def reference(
        mesh,
        **kwargs,
    ) -> tuple[np.array]:
        z_w_con_c_full = 0.
        ddt_w_adv = 0.
        return dict(z_w_con_c_full=z_w_con_c_full, ddt_w_adv=ddt_w_adv)

    @pytest.fixture
    def input_data(self, mesh):

        z_w_con_c = random_field(mesh, CellDim, KDim, extend={KDim: 1})
        w = random_field(mesh, CellDim, KDim, extend={KDim: 1})
        coeff1_dwdz = random_field(mesh, CellDim, KDim)
        coeff2_dwdz = random_field(mesh, CellDim, KDim)

        z_v_grad_w = random_field(mesh, EdgeDim, KDim)
        e_bln_c_s = random_field(mesh, CellDim, C2EDim)

        levelmask = random_mask(mesh, KDim, extend={KDim: 1})
        cfl_clipping = random_mask(mesh, CellDim, KDim)
        owner_mask = random_mask(mesh, CellDim)
        ddqz_z_half = random_field(mesh, CellDim, KDim)
        area = random_field(mesh, CellDim)
        geofac_n2s = random_field(mesh, CellDim, C2E2CODim)

        z_w_con_c_full = zero_field(mesh, CellDim, KDim)
        ddt_w_adv = zero_field(mesh, CellDim, KDim)

        scalfac_exdiff = 10.0
        cfl_w_limit = 3.0
        dtime = 2.0

        vert_idx = zero_field(mesh, KDim, dtype=int32)
        for level in range(mesh.k_level):
            vert_idx[level] = level

        horz_idx = zero_field(mesh, CellDim, dtype=int32)
        for cell in range(mesh.n_cells):
            horz_idx[cell] = cell

        nlev = mesh.k_level
        nrdmax = 5
        extra_diffu = True

        horz_lower_bound=2
        horz_upper_bound=4

        lvn_only = False

        return dict(
            z_w_con_c=z_w_con_c,
            w=w,
            coeff1_dwdz=coeff1_dwdz,
            coeff2_dwdz=coeff2_dwdz,
            ddt_w_adv=ddt_w_adv,
            e_bln_c_s=e_bln_c_s,
            z_v_grad_w=z_v_grad_w,
            levelmask=levelmask,
            cfl_clipping=cfl_clipping,
            owner_mask=owner_mask,
            ddqz_z_half=ddqz_z_half,
            area=area,
            geofac_n2s=geofac_n2s,
            horz_idx=horz_idx,
            vert_idx=vert_idx,
            scalfac_exdiff=scalfac_exdiff,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            horz_lower_bound=horz_lower_bound,
            horz_upper_bound=horz_upper_bound,
            nlev=nlev,
            nrdmax=nrdmax,
            lvn_only=lvn_only,
            extra_diffu=extra_diffu,
            z_w_con_c_full=z_w_con_c_full,
        )
