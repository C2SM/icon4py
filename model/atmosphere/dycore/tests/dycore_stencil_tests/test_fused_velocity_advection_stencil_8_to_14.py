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

from icon4py.model.atmosphere.dycore.fused_velocity_advection_stencil_8_to_14 import (
    fused_velocity_advection_stencil_8_to_14,
)
from icon4py.model.common.dimension import C2EDim, CEDim, CellDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    as_1D_sparse_field,
    random_field,
    random_mask,
    zero_field,
)

from .test_interpolate_to_cell_center import interpolate_to_cell_center_numpy
from .test_mo_velocity_advection_stencil_10 import mo_velocity_advection_stencil_10_numpy
from .test_mo_velocity_advection_stencil_11 import mo_velocity_advection_stencil_11_numpy
from .test_mo_velocity_advection_stencil_12 import mo_velocity_advection_stencil_12_numpy
from .test_mo_velocity_advection_stencil_13 import mo_velocity_advection_stencil_13_numpy
from .test_mo_velocity_advection_stencil_14 import mo_velocity_advection_stencil_14_numpy


class TestFusedVelocityAdvectionStencil8To14(StencilTest):
    PROGRAM = fused_velocity_advection_stencil_8_to_14
    OUTPUTS = (
        "z_ekinh",
        "cfl_clipping",
        "pre_levelmask",
        "vcfl",
        "z_w_con_c",
    )

    @staticmethod
    def reference(
        grid,
        z_kin_hor_e,
        e_bln_c_s,
        z_w_concorr_me,
        wgtfac_c,
        w,
        ddqz_z_half,
        cfl_clipping,
        pre_levelmask,
        vcfl,
        z_w_concorr_mc,
        w_concorr_c,
        z_ekinh,
        k,
        istep,
        cfl_w_limit,
        dtime,
        nlevp1,
        nlev,
        nflatlev,
        nrdmax,
        z_w_con_c,
        **kwargs,
    ):

        z_ekinh = np.where(
            k < nlev,
            interpolate_to_cell_center_numpy(grid, z_kin_hor_e, e_bln_c_s),
            z_ekinh,
        )

        if istep == 1:
            z_w_concorr_mc = np.where(
                (nflatlev < k) & (k < nlev),
                interpolate_to_cell_center_numpy(grid, z_w_concorr_me, e_bln_c_s),
                z_w_concorr_mc,
            )

            w_concorr_c = np.where(
                (nflatlev + 1 < k) & (k < nlev),
                mo_velocity_advection_stencil_10_numpy(grid, z_w_concorr_mc, wgtfac_c),
                w_concorr_c,
            )

        z_w_con_c = np.where(
            k < nlevp1,
            mo_velocity_advection_stencil_11_numpy(w),
            mo_velocity_advection_stencil_12_numpy(z_w_con_c),
        )

        z_w_con_c = np.where(
            (nflatlev + 1 < k) & (k < nlev),
            mo_velocity_advection_stencil_13_numpy(z_w_con_c, w_concorr_c),
            z_w_con_c,
        )

        condition = (np.maximum(3, nrdmax - 2) < k) & (k < nlev - 3)
        cfl_clipping_new, vcfl_new, z_w_con_c_new = mo_velocity_advection_stencil_14_numpy(
            grid, ddqz_z_half, z_w_con_c, cfl_w_limit, dtime
        )

        cfl_clipping = np.where(condition, cfl_clipping_new, cfl_clipping)
        vcfl = np.where(condition, vcfl_new, vcfl)
        z_w_con_c = np.where(condition, z_w_con_c_new, z_w_con_c)

        return dict(
            z_ekinh=z_ekinh,
            cfl_clipping=cfl_clipping,
            pre_levelmask=pre_levelmask,
            vcfl=vcfl,
            z_w_con_c=z_w_con_c,
        )

    @pytest.fixture
    def input_data(self, grid):
        z_kin_hor_e = random_field(grid, EdgeDim, KDim)
        e_bln_c_s = random_field(grid, CellDim, C2EDim)
        z_ekinh = zero_field(grid, CellDim, KDim)
        z_w_concorr_me = random_field(grid, EdgeDim, KDim)
        z_w_concorr_mc = zero_field(grid, CellDim, KDim)
        wgtfac_c = random_field(grid, CellDim, KDim)
        w_concorr_c = zero_field(grid, CellDim, KDim)
        w = random_field(grid, CellDim, KDim)
        z_w_con_c = zero_field(grid, CellDim, KDim)
        ddqz_z_half = random_field(grid, CellDim, KDim)
        cfl_clipping = random_mask(grid, CellDim, KDim, dtype=bool)
        pre_levelmask = random_mask(
            grid, CellDim, KDim, dtype=bool
        )  # TODO should be just a K field

        vcfl = zero_field(grid, CellDim, KDim)
        cfl_w_limit = 5.0
        dtime = 9.0

        k = zero_field(grid, KDim, dtype=int32)
        for level in range(grid.num_levels):
            k[level] = level

        nlevp1 = grid.num_levels + 1
        nlev = grid.num_levels
        nflatlev = 13
        nrdmax = 10

        istep = 1
        return dict(
            z_kin_hor_e=z_kin_hor_e,
            e_bln_c_s=as_1D_sparse_field(e_bln_c_s, CEDim),
            z_w_concorr_me=z_w_concorr_me,
            wgtfac_c=wgtfac_c,
            w=w,
            ddqz_z_half=ddqz_z_half,
            cfl_clipping=cfl_clipping,
            pre_levelmask=pre_levelmask,
            vcfl=vcfl,
            z_w_concorr_mc=z_w_concorr_mc,
            w_concorr_c=w_concorr_c,
            z_ekinh=z_ekinh,
            k=k,
            istep=istep,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            nlevp1=nlevp1,
            nlev=nlev,
            nflatlev=nflatlev,
            nrdmax=nrdmax,
            z_w_con_c=z_w_con_c,
        )
