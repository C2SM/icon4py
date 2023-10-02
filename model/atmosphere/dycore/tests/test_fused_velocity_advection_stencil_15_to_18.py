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

from icon4py.model.atmosphere.dycore.fused_velocity_advection_stencil_15_to_18 import (
    fused_velocity_advection_stencil_15_to_18,
)
from icon4py.model.common.dimension import C2E2CODim, C2EDim, CellDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    random_field,
    random_mask,
    zero_field,
)

from .test_mo_velocity_advection_stencil_15 import TestMoVelocityAdvectionStencil15
from .test_mo_velocity_advection_stencil_16 import mo_velocity_advection_stencil_16_numpy
from .test_mo_velocity_advection_stencil_17 import TestMoVelocityAdvectionStencil17
from .test_mo_velocity_advection_stencil_18 import TestMoVelocityAdvectionStencil18


_mo_velocity_advection_stencil_15 = TestMoVelocityAdvectionStencil15.reference
_mo_velocity_advection_stencil_16 = mo_velocity_advection_stencil_16_numpy
_mo_velocity_advection_stencil_17 = TestMoVelocityAdvectionStencil17.reference
_mo_velocity_advection_stencil_18 = TestMoVelocityAdvectionStencil18.reference


class TestFusedVelocityAdvectionStencil15To18(StencilTest):
    PROGRAM = fused_velocity_advection_stencil_15_to_18
    OUTPUTS = (
        "z_w_con_c_full",
        "ddt_w_adv",
    )

    @staticmethod
    def _fused_velocity_advection_stencil_16_to_18(
        mesh,
        z_w_con_c,
        w,
        coeff1_dwdz,
        coeff2_dwdz,
        ddt_w_adv,
        e_bln_c_s,
        z_v_grad_w,
        levelmask,
        cfl_clipping,
        owner_mask,
        ddqz_z_half,
        area,
        geofac_n2s,
        horz_idx,
        vert_idx,
        scalfac_exdiff,
        cfl_w_limit,
        dtime,
        horz_lower_bound,
        horz_upper_bound,
        nlev,
        nrdmax,
        extra_diffu,
    ):
        horz_idx = horz_idx[:, np.newaxis]
        vert_idx = vert_idx[np.newaxis, :]
        condition1 = (horz_lower_bound < horz_idx) & (horz_idx < horz_upper_bound) & (vert_idx > 0)

        ddt_w_adv = np.where(
            condition1,
            _mo_velocity_advection_stencil_16(z_w_con_c, w, coeff1_dwdz, coeff2_dwdz),
            ddt_w_adv,
        )

        ddt_w_adv = np.where(
            condition1,
            _mo_velocity_advection_stencil_17(mesh, e_bln_c_s, z_v_grad_w, ddt_w_adv),
            ddt_w_adv,
        )

        condition2 = (
            (horz_lower_bound < horz_idx)
            & (horz_idx < horz_upper_bound)
            & (np.maximum(3, nrdmax - 2) < vert_idx)
            & (vert_idx < nlev - 4)
        )

        if extra_diffu:
            ddt_w_adv = np.where(
                condition2,
                _mo_velocity_advection_stencil_18(
                    mesh,
                    levelmask,
                    cfl_clipping,
                    owner_mask,
                    z_w_con_c,
                    ddqz_z_half,
                    area,
                    geofac_n2s,
                    w,
                    ddt_w_adv,
                    scalfac_exdiff,
                    cfl_w_limit,
                    dtime,
                ),
                ddt_w_adv,
            )
        return ddt_w_adv

    @classmethod
    def reference(
        cls,
        mesh,
        z_w_con_c,
        w,
        coeff1_dwdz,
        coeff2_dwdz,
        ddt_w_adv,
        e_bln_c_s,
        z_v_grad_w,
        levelmask,
        cfl_clipping,
        owner_mask,
        ddqz_z_half,
        area,
        geofac_n2s,
        horz_idx,
        vert_idx,
        scalfac_exdiff,
        cfl_w_limit,
        dtime,
        horz_lower_bound,
        horz_upper_bound,
        nlev,
        nrdmax,
        lvn_only,
        extra_diffu,
        **kwargs,
    ) -> dict:
        z_w_con_c_full = _mo_velocity_advection_stencil_15(mesh, z_w_con_c)

        if not lvn_only:
            ddt_w_adv = cls._fused_velocity_advection_stencil_16_to_18(
                mesh,
                z_w_con_c,
                w,
                coeff1_dwdz,
                coeff2_dwdz,
                ddt_w_adv,
                e_bln_c_s,
                z_v_grad_w,
                levelmask,
                cfl_clipping,
                owner_mask,
                ddqz_z_half,
                area,
                geofac_n2s,
                horz_idx,
                vert_idx,
                scalfac_exdiff,
                cfl_w_limit,
                dtime,
                horz_lower_bound,
                horz_upper_bound,
                nlev,
                nrdmax,
                extra_diffu,
            )

        return dict(z_w_con_c_full=z_w_con_c_full, ddt_w_adv=ddt_w_adv)

    @pytest.fixture
    def input_data(self, mesh):
        z_w_con_c = random_field(mesh, CellDim, KDim)
        w = random_field(mesh, CellDim, KDim, extend={KDim: 1})
        coeff1_dwdz = random_field(mesh, CellDim, KDim)
        coeff2_dwdz = random_field(mesh, CellDim, KDim)

        z_v_grad_w = random_field(mesh, EdgeDim, KDim)
        e_bln_c_s = random_field(mesh, CellDim, C2EDim)

        levelmask = random_mask(mesh, KDim)
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

        horz_lower_bound = 2
        horz_upper_bound = 4

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
