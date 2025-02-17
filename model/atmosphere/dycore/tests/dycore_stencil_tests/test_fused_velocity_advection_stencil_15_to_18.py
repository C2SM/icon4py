# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import numpy as np
import pytest

import icon4py.model.common.utils.data_allocation as data_alloc
from icon4py.model.atmosphere.dycore.stencils.fused_velocity_advection_stencil_15_to_18 import (
    fused_velocity_advection_stencil_15_to_18,
)
from icon4py.model.common import dimension as dims
from icon4py.model.testing.helpers import StencilTest

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


def _fused_velocity_advection_stencil_16_to_18(
    connectivities: dict[gtx.Dimension, np.ndarray],
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
            connectivities, e_bln_c_s, z_v_grad_w, ddt_w_adv
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
                connectivities,
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


class TestFusedVelocityAdvectionStencil15To18(StencilTest):
    PROGRAM = fused_velocity_advection_stencil_15_to_18
    OUTPUTS = (
        "z_w_con_c_full",
        "ddt_w_adv",
    )
    MARKERS = (pytest.mark.embedded_remap_error,)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
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
        z_w_con_c_full = interpolate_contravariant_vertical_velocity_to_full_levels_numpy(z_w_con_c)

        if not lvn_only:
            ddt_w_adv = _fused_velocity_advection_stencil_16_to_18(
                connectivities,
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
        z_w_con_c = data_alloc.random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        w = data_alloc.random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        coeff1_dwdz = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        coeff2_dwdz = data_alloc.random_field(grid, dims.CellDim, dims.KDim)

        z_v_grad_w = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        e_bln_c_s = data_alloc.random_field(grid, dims.CEDim)

        levelmask = data_alloc.random_mask(grid, dims.KDim)
        cfl_clipping = data_alloc.random_mask(grid, dims.CellDim, dims.KDim)
        owner_mask = data_alloc.random_mask(grid, dims.CellDim)
        ddqz_z_half = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        area = data_alloc.random_field(grid, dims.CellDim)
        geofac_n2s = data_alloc.random_field(grid, dims.CellDim, dims.C2E2CODim)

        z_w_con_c_full = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        ddt_w_adv = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)

        scalfac_exdiff = 10.0
        cfl_w_limit = 3.0
        dtime = 2.0

        k = data_alloc.zero_field(grid, dims.KDim, dtype=gtx.int32)
        for level in range(grid.num_levels):
            k[level] = level

        cell = data_alloc.zero_field(grid, dims.CellDim, dtype=gtx.int32)
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
