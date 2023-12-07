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

from icon4py.model.atmosphere.diffusion.stencils.apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence import (
    apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence,
)
from icon4py.model.common.dimension import C2E2CODim, CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field

from .test_apply_nabla2_to_w import apply_nabla2_to_w_numpy
from .test_apply_nabla2_to_w_in_upper_damping_layer import (
    apply_nabla2_to_w_in_upper_damping_layer_numpy,
)
from .test_calculate_horizontal_gradients_for_turbulence import (
    calculate_horizontal_gradients_for_turbulence_numpy,
)
from .test_calculate_nabla2_for_w import calculate_nabla2_for_w_numpy


class TestApplyDiffusionToWAndComputeHorizontalGradientsForTurbulence(StencilTest):
    PROGRAM = apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence
    OUTPUTS = ("w", "dwdx", "dwdy")

    @staticmethod
    def reference(
        grid,
        area,
        geofac_n2s,
        geofac_grg_x,
        geofac_grg_y,
        w_old,
        dwdx,
        dwdy,
        diff_multfac_w,
        diff_multfac_n2w,
        k,
        cell,
        nrdmax,
        interior_idx,
        halo_idx,
        **kwargs,
    ):
        reshaped_k = k[np.newaxis, :]
        reshaped_cell = cell[:, np.newaxis]

        dwdx, dwdy = np.where(
            0 < reshaped_k,
            calculate_horizontal_gradients_for_turbulence_numpy(
                grid, w_old, geofac_grg_x, geofac_grg_y
            ),
            (dwdx, dwdy),
        )

        z_nabla2_c = calculate_nabla2_for_w_numpy(grid, w_old, geofac_n2s)

        w = np.where(
            (interior_idx <= reshaped_cell) & (reshaped_cell < halo_idx),
            apply_nabla2_to_w_numpy(grid, area, z_nabla2_c, geofac_n2s, w_old, diff_multfac_w),
            w_old,
        )

        w = np.where(
            (0 < reshaped_k)
            & (reshaped_k < nrdmax)
            & (interior_idx <= reshaped_cell)
            & (reshaped_cell < halo_idx),
            apply_nabla2_to_w_in_upper_damping_layer_numpy(w, diff_multfac_n2w, area, z_nabla2_c),
            w,
        )
        return dict(w=w, dwdx=dwdx, dwdy=dwdy)

    @pytest.fixture
    def input_data(self, grid):
        k = zero_field(grid, KDim, dtype=int32)
        for lev in range(grid.num_levels):
            k[lev] = lev

        cell = zero_field(grid, CellDim, dtype=int32)
        for c in range(grid.num_cells):
            cell[c] = c

        nrdmax = 13
        interior_idx = 1
        halo_idx = 5

        geofac_grg_x = random_field(grid, CellDim, C2E2CODim)
        geofac_grg_y = random_field(grid, CellDim, C2E2CODim)
        diff_multfac_n2w = random_field(grid, KDim)
        area = random_field(grid, CellDim)
        geofac_n2s = random_field(grid, CellDim, C2E2CODim)
        w_old = random_field(grid, CellDim, KDim)
        diff_multfac_w = 5.0

        w = zero_field(grid, CellDim, KDim)
        dwdx = zero_field(grid, CellDim, KDim)
        dwdy = zero_field(grid, CellDim, KDim)

        return dict(
            area=area,
            geofac_n2s=geofac_n2s,
            geofac_grg_x=geofac_grg_x,
            geofac_grg_y=geofac_grg_y,
            w_old=w_old,
            diff_multfac_w=diff_multfac_w,
            diff_multfac_n2w=diff_multfac_n2w,
            k=k,
            cell=cell,
            nrdmax=nrdmax,
            interior_idx=interior_idx,
            halo_idx=halo_idx,
            w=w,
            dwdx=dwdx,
            dwdy=dwdy,
            horizontal_start=0,
            horizontal_end=grid.num_cells,
            vertical_start=0,
            vertical_end=grid.num_levels,
        )
