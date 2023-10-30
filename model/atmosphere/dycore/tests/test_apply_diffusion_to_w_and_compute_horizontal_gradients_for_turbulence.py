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

from icon4py.model.atmosphere.dycore.apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence import (
    apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence,
)
from icon4py.model.common.dimension import C2E2CODim, CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field


class TestApplyDiffusionToWAndComputeHorizontalGradientsForTurbulence(StencilTest):
    PROGRAM = apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence
    OUTPUTS = ("w", "dwdx", "dwdy")

    @staticmethod
    def reference(
        mesh,
        area,
        geofac_n2s,
        geofac_grg_x,
        geofac_grg_y,
        w_old,
        diff_multfac_w,
        diff_multfac_n2w,
        vert_idx,
        horz_idx,
        nrdmax,
        interior_idx,
        halo_idx,
        w,
        dwdx,
        dwdy,
    ) -> np.array:

        w = 0.0
        dwdx = 0.0
        dwdy = 0.0

        return dict(w=w, dwdx=dwdx, dwdy=dwdy)

    @pytest.fixture
    def input_data(self, mesh):

        vert_idx = zero_field(mesh, KDim, dtype=int32)
        for k in range(mesh.k_level):
            vert_idx[k] = k

        horz_idx = zero_field(mesh, CellDim, dtype=int32)
        for cell in range(mesh.n_cells):
            horz_idx[cell] = cell

        nrdmax = 13
        interior_idx = 1
        halo_idx = 5

        geofac_grg_x = random_field(mesh, CellDim, C2E2CODim)
        geofac_grg_y = random_field(mesh, CellDim, C2E2CODim)
        diff_multfac_n2w = random_field(mesh, KDim)
        area = random_field(mesh, CellDim)
        geofac_n2s = random_field(mesh, CellDim, C2E2CODim)
        w_old = random_field(mesh, CellDim, KDim)
        diff_multfac_w = 5.0

        w = zero_field(mesh, CellDim, KDim)
        dwdx = zero_field(mesh, CellDim, KDim)
        dwdy = zero_field(mesh, CellDim, KDim)

        return dict(
            area=area,
            geofac_n2s=geofac_n2s,
            geofac_grg_x=geofac_grg_x,
            geofac_grg_y=geofac_grg_y,
            w_old=w_old,
            diff_multfac_w=diff_multfac_w,
            diff_multfac_n2w=diff_multfac_n2w,
            vert_idx=vert_idx,
            horz_idx=horz_idx,
            nrdmax=nrdmax,
            interior_idx=interior_idx,
            halo_idx=halo_idx,
            w=w,
            dwdx=dwdx,
            dwdy=dwdy,
        )
