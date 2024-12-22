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

from icon4py.model.atmosphere.dycore.stencils.mo_solve_nonhydro_stencil_51 import (
    mo_solve_nonhydro_stencil_51,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.utils.data_allocation import random_field, zero_field
from icon4py.model.testing.helpers import StencilTest


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
        z_q = zero_field(grid, dims.CellDim, dims.KDim)
        w_nnew = zero_field(grid, dims.CellDim, dims.KDim)
        vwind_impl_wgt = random_field(grid, dims.CellDim)
        theta_v_ic = random_field(grid, dims.CellDim, dims.KDim)
        ddqz_z_half = random_field(grid, dims.CellDim, dims.KDim, low=0.5, high=1.5)
        z_beta = random_field(grid, dims.CellDim, dims.KDim, low=0.5, high=1.5)
        z_alpha = random_field(
            grid, dims.CellDim, dims.KDim, low=0.5, high=1.5, extend={dims.KDim: 1}
        )
        z_w_expl = random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        z_exner_expl = random_field(grid, dims.CellDim, dims.KDim)
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
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=1,
            vertical_end=gtx.int32(grid.num_levels),
        )
