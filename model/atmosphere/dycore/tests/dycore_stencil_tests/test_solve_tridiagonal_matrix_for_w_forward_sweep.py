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

from icon4py.model.atmosphere.dycore.stencils.solve_tridiagonal_matrix_for_w_forward_sweep import (
    solve_tridiagonal_matrix_for_w_forward_sweep,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base as base_grid
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import random_field
from icon4py.model.testing.helpers import StencilTest


def solve_tridiagonal_matrix_for_w_forward_sweep_numpy(
    vwind_impl_wgt: np.array,
    theta_v_ic: np.array,
    ddqz_z_half: np.array,
    z_alpha: np.array,
    z_beta: np.array,
    z_exner_expl: np.array,
    z_w_expl: np.array,
    z_q_ref: np.array,
    w_ref: np.array,
    dtime,
    cpd,
) -> tuple[np.array]:
    z_q = np.copy(z_q_ref)
    w = np.copy(w_ref)
    vwind_impl_wgt = np.expand_dims(vwind_impl_wgt, axis=-1)

    z_gamma = dtime * cpd * vwind_impl_wgt * theta_v_ic / ddqz_z_half
    z_a = np.zeros_like(z_gamma)
    z_b = np.zeros_like(z_gamma)
    z_c = np.zeros_like(z_gamma)
    z_g = np.zeros_like(z_gamma)

    k_size = w.shape[1]
    for k in range(1, k_size):
        z_a[:, k] = -z_gamma[:, k] * z_beta[:, k - 1] * z_alpha[:, k - 1]
        z_c[:, k] = -z_gamma[:, k] * z_beta[:, k] * z_alpha[:, k + 1]
        z_b[:, k] = 1.0 + z_gamma[:, k] * z_alpha[:, k] * (z_beta[:, k - 1] + z_beta[:, k])
        z_g[:, k] = 1.0 / (z_b[:, k] + z_a[:, k] * z_q[:, k - 1])
        z_q[:, k] = -z_c[:, k] * z_g[:, k]

        w[:, k] = z_w_expl[:, k] - z_gamma[:, k] * (z_exner_expl[:, k - 1] - z_exner_expl[:, k])
        w[:, k] = (w[:, k] - z_a[:, k] * w[:, k - 1]) * z_g[:, k]
    return z_q, w


class TestSolveTridiagonalMatrixForWForwardSweep(StencilTest):
    PROGRAM = solve_tridiagonal_matrix_for_w_forward_sweep
    OUTPUTS = ("w", "z_q")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        vwind_impl_wgt: np.ndarray,
        theta_v_ic: np.ndarray,
        ddqz_z_half: np.ndarray,
        z_alpha: np.ndarray,
        z_beta: np.ndarray,
        z_w_expl: np.ndarray,
        z_exner_expl: np.ndarray,
        z_q: np.ndarray,
        w: np.ndarray,
        dtime: wpfloat,
        cpd: wpfloat,
        **kwargs,
    ) -> dict:
        z_q_ref, w_ref = solve_tridiagonal_matrix_for_w_forward_sweep_numpy(
            vwind_impl_wgt,
            theta_v_ic,
            ddqz_z_half,
            z_alpha,
            z_beta,
            z_exner_expl,
            z_w_expl,
            z_q,
            w,
            dtime,
            cpd,
        )
        return dict(z_q=z_q_ref, w=w_ref)

    @pytest.fixture
    def input_data(self, grid: base_grid.BaseGrid):
        vwind_impl_wgt = random_field(grid, dims.CellDim, dtype=wpfloat)
        theta_v_ic = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        ddqz_z_half = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        z_alpha = random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=vpfloat)
        z_beta = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        z_exner_expl = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        z_w_expl = random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=wpfloat)
        z_q = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        w = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)

        h_start = 0
        h_end = gtx.int32(grid.num_cells)
        v_start = 1
        v_end = gtx.int32(grid.num_levels)

        return dict(
            vwind_impl_wgt=vwind_impl_wgt,
            theta_v_ic=theta_v_ic,
            ddqz_z_half=ddqz_z_half,
            z_alpha=z_alpha,
            z_beta=z_beta,
            z_w_expl=z_w_expl,
            z_exner_expl=z_exner_expl,
            z_q=z_q,
            w=w,
            dtime=wpfloat("8.0"),
            cpd=wpfloat("7.0"),
            horizontal_start=h_start,
            horizontal_end=h_end,
            vertical_start=v_start,
            vertical_end=v_end,
        )
