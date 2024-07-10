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
from icon4py.model.common.dimension import C2E2CODim, C2EDim, CEDim, CellDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    as_1D_sparse_field,
    random_field,
    random_mask,
    zero_field,
)

from .test_add_extra_diffusion_for_w_con_approaching_cfl import (
    add_extra_diffusion_for_w_con_approaching_cfl_numpy,
)
from .test_add_interpolated_horizontal_advection_of_w import (
    add_interpolated_horizontal_advection_of_w_numpy,
)
from .test_compute_advective_vertical_wind_tendency import (
    compute_advective_vertical_wind_tendency_numpy,
)
from .test_interpolate_contravariant_vertical_velocity_to_full_levels import (
    interpolate_contravariant_vertical_velocity_to_full_levels_numpy,
)


class TestFusedVelocityAdvectionStencil15To18(StencilTest):
    PROGRAM = fused_velocity_advection_stencil_15_to_18
    OUTPUTS = (
        "z_w_con_c_full",
        "ddt_w_adv",
    )

    @staticmethod
    def _fused_velocity_advection_stencil_16_to_18(
        grid,
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
        cell,
        k,
        scalfac_exdiff,
        cfl_w_limit,
        dtime,
        cell_lower_bound,
        cell_upper_bound,
        nlev,
        nrdmax,
        extra_diffu,
    ):
        cell = cell[:, np.newaxis]

        condition1 = (cell_lower_bound <= cell) & (cell < cell_upper_bound) & (k >= 1)

        ddt_w_adv = np.where(
            condition1,
            compute_advective_vertical_wind_tendency_numpy(
                z_w_con_c[:, :-1], w, coeff1_dwdz, coeff2_dwdz
            ),
            ddt_w_adv,
        )

        ddt_w_adv = np.where(
            condition1,
            add_interpolated_horizontal_advection_of_w_numpy(
                grid, e_bln_c_s, z_v_grad_w, ddt_w_adv
            ),
            ddt_w_adv,
        )

        condition2 = (
            (cell_lower_bound <= cell)
            & (cell < cell_upper_bound)
            & (np.maximum(2, nrdmax - 2) <= k)
            & (k < nlev - 3)
        )

        if extra_diffu:
            ddt_w_adv = np.where(
                condition2,
                add_extra_diffusion_for_w_con_approaching_cfl_numpy(
                    grid,
                    levelmask,
                    cfl_clipping,
                    owner_mask,
                    z_w_con_c[:, :-1],
                    ddqz_z_half,
                    area,
                    geofac_n2s,
                    w[:, :-1],
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
        grid,
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
        cell,
        k,
        scalfac_exdiff,
        cfl_w_limit,
        dtime,
        cell_lower_bound,
        cell_upper_bound,
        nlev,
        nrdmax,
        lvn_only,
        extra_diffu,
        **kwargs,
    ):
        z_w_con_c_full = interpolate_contravariant_vertical_velocity_to_full_levels_numpy(
            grid, z_w_con_c
        )

        if not lvn_only:
            ddt_w_adv = cls._fused_velocity_advection_stencil_16_to_18(
                grid,
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
                cell,
                k,
                scalfac_exdiff,
                cfl_w_limit,
                dtime,
                cell_lower_bound,
                cell_upper_bound,
                nlev,
                nrdmax,
                extra_diffu,
            )

        return dict(z_w_con_c_full=z_w_con_c_full, ddt_w_adv=ddt_w_adv)

    @pytest.fixture
    def input_data(self, grid):
        z_w_con_c = random_field(grid, CellDim, KDim, extend={KDim: 1})
        w = random_field(grid, CellDim, KDim, extend={KDim: 1})
        coeff1_dwdz = random_field(grid, CellDim, KDim)
        coeff2_dwdz = random_field(grid, CellDim, KDim)

        z_v_grad_w = random_field(grid, EdgeDim, KDim)
        e_bln_c_s = as_1D_sparse_field(random_field(grid, CellDim, C2EDim), CEDim)

        levelmask = random_mask(grid, KDim)
        cfl_clipping = random_mask(grid, CellDim, KDim)
        owner_mask = random_mask(grid, CellDim)
        ddqz_z_half = random_field(grid, CellDim, KDim)
        area = random_field(grid, CellDim)
        geofac_n2s = random_field(grid, CellDim, C2E2CODim)

        z_w_con_c_full = zero_field(grid, CellDim, KDim)
        ddt_w_adv = zero_field(grid, CellDim, KDim)

        scalfac_exdiff = 10.0
        cfl_w_limit = 3.0
        dtime = 2.0

        k = zero_field(grid, KDim, dtype=int32)
        for level in range(grid.num_levels):
            k[level] = level

        cell = zero_field(grid, CellDim, dtype=int32)
        for c in range(grid.num_cells):
            cell[c] = c

        nlev = grid.num_levels
        nrdmax = 5
        extra_diffu = True

        cell_lower_bound = 2
        cell_upper_bound = 4

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
            cell=cell,
            k=k,
            scalfac_exdiff=scalfac_exdiff,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            cell_lower_bound=cell_lower_bound,
            cell_upper_bound=cell_upper_bound,
            nlev=nlev,
            nrdmax=nrdmax,
            lvn_only=lvn_only,
            extra_diffu=extra_diffu,
            z_w_con_c_full=z_w_con_c_full,
        )
