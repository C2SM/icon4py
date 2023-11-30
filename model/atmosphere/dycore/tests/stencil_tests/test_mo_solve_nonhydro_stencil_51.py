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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_51 import (
    mo_solve_nonhydro_stencil_51,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field


def mo_solve_nonhydro_stencil_51_z_q_numpy(
    z_c: np.array,
    z_b: np.array,
) -> np.array:
    return -z_c / z_b


def mo_solve_nonhydro_stencil_51_w_nnew_numpy(
    z_gamma: np.array,
    z_b: np.array,
    z_w_expl: np.array,
    z_exner_expl: np.array,
) -> np.array:
    z_exner_expl_k_minus_1 = np.roll(z_exner_expl, shift=1, axis=1)
    w_nnew = z_w_expl[:, :-1] - z_gamma * (z_exner_expl_k_minus_1 - z_exner_expl)
    return w_nnew / z_b


class TestMoSolveNonHydroStencil51(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_51
    OUTPUTS = ("z_q", "w_nnew")

    @staticmethod
    def reference(
        grid,
        vwind_impl_wgt: np.array,
        theta_v_ic: np.array,
        ddqz_z_half: np.array,
        z_beta: np.array,
        z_alpha: np.array,
        z_w_expl: np.array,
        z_exner_expl: np.array,
        dtime: float,
        cpd: float,
        **kwargs,
    ) -> dict:
        vwind_impl_wgt = np.expand_dims(vwind_impl_wgt, axis=-1)
        z_gamma = dtime * cpd * vwind_impl_wgt * theta_v_ic / ddqz_z_half
        z_alpha_k_plus_1 = z_alpha[:, 1:]
        z_beta_k_minus_1 = np.roll(z_beta, shift=1, axis=1)
        z_c = -z_gamma * z_beta * z_alpha_k_plus_1
        z_b = 1.0 + z_gamma * z_alpha[:, :-1] * (z_beta_k_minus_1 + z_beta)
        z_q = np.zeros_like(z_b)

        z_q[:, 1:] = mo_solve_nonhydro_stencil_51_z_q_numpy(z_c, z_b)[:, 1:]

        w_nnew = np.zeros_like(z_q)
        w_nnew[:, 1:] = mo_solve_nonhydro_stencil_51_w_nnew_numpy(
            z_gamma, z_b, z_w_expl, z_exner_expl
        )[:, 1:]

        return dict(z_q=z_q, w_nnew=w_nnew)

    @pytest.fixture
    def input_data(self, grid):
        z_q = zero_field(grid, CellDim, KDim)
        w_nnew = zero_field(grid, CellDim, KDim)
        vwind_impl_wgt = random_field(grid, CellDim)
        theta_v_ic = random_field(grid, CellDim, KDim)
        ddqz_z_half = random_field(grid, CellDim, KDim, low=0.5, high=1.5)
        z_beta = random_field(grid, CellDim, KDim, low=0.5, high=1.5)
        z_alpha = random_field(grid, CellDim, KDim, low=0.5, high=1.5, extend={KDim: 1})
        z_w_expl = random_field(grid, CellDim, KDim, extend={KDim: 1})
        z_exner_expl = random_field(grid, CellDim, KDim)
        dtime = 10.0
        cpd = 1.0
        return dict(
            z_q=z_q,
            w_nnew=w_nnew,
            vwind_impl_wgt=vwind_impl_wgt,
            theta_v_ic=theta_v_ic,
            ddqz_z_half=ddqz_z_half,
            z_beta=z_beta,
            z_alpha=z_alpha,
            z_w_expl=z_w_expl,
            z_exner_expl=z_exner_expl,
            dtime=dtime,
            cpd=cpd,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_cells),
            vertical_start=int32(1),
            vertical_end=int32(grid.num_levels),
        )
