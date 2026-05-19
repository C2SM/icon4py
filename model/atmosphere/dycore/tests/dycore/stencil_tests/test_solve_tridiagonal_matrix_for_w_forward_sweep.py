# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.dycore.stencils.solve_tridiagonal_matrix_for_w_forward_sweep import (
    solve_tridiagonal_matrix_for_w_forward_sweep,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base as base_grid
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


def solve_tridiagonal_matrix_for_w_forward_sweep_numpy(
    exner_w_implicit_weight_parameter: np.ndarray,
    theta_v_at_cells_on_half_levels: np.ndarray,
    ddqz_z_half: np.ndarray,
    tridiagonal_alpha_coeff_at_cells_on_half_levels: np.ndarray,
    tridiagonal_beta_coeff_at_cells_on_model_levels: np.ndarray,
    exner_explicit_term: np.ndarray,
    w_explicit_term: np.ndarray,
    z_q_ref: np.ndarray,
    w_ref: np.ndarray,
    dtime: float,
    cpd: float,
) -> tuple[np.ndarray, np.ndarray]:
    tridiagonal_intermediate_result = np.copy(z_q_ref)
    w = np.copy(w_ref)
    exner_w_implicit_weight_parameter = np.expand_dims(exner_w_implicit_weight_parameter, axis=-1)

    z_gamma = (
        dtime
        * cpd
        * exner_w_implicit_weight_parameter
        * theta_v_at_cells_on_half_levels
        / ddqz_z_half
    )
    z_a = np.zeros_like(z_gamma)
    z_b = np.zeros_like(z_gamma)
    z_c = np.zeros_like(z_gamma)
    z_g = np.zeros_like(z_gamma)

    k_size = w.shape[1]
    for k in range(1, k_size):
        z_a[:, k] = (
            -z_gamma[:, k]
            * tridiagonal_beta_coeff_at_cells_on_model_levels[:, k - 1]
            * tridiagonal_alpha_coeff_at_cells_on_half_levels[:, k - 1]
        )
        z_c[:, k] = (
            -z_gamma[:, k]
            * tridiagonal_beta_coeff_at_cells_on_model_levels[:, k]
            * tridiagonal_alpha_coeff_at_cells_on_half_levels[:, k + 1]
        )
        z_b[:, k] = 1.0 + z_gamma[:, k] * tridiagonal_alpha_coeff_at_cells_on_half_levels[:, k] * (
            tridiagonal_beta_coeff_at_cells_on_model_levels[:, k - 1]
            + tridiagonal_beta_coeff_at_cells_on_model_levels[:, k]
        )
        z_g[:, k] = 1.0 / (z_b[:, k] + z_a[:, k] * tridiagonal_intermediate_result[:, k - 1])
        tridiagonal_intermediate_result[:, k] = -z_c[:, k] * z_g[:, k]

        w[:, k] = w_explicit_term[:, k] - z_gamma[:, k] * (
            exner_explicit_term[:, k - 1] - exner_explicit_term[:, k]
        )
        w[:, k] = (w[:, k] - z_a[:, k] * w[:, k - 1]) * z_g[:, k]
    return tridiagonal_intermediate_result, w


class TestSolveTridiagonalMatrixForWForwardSweep(StencilTest):
    PROGRAM = solve_tridiagonal_matrix_for_w_forward_sweep
    OUTPUTS = ("w", "tridiagonal_intermediate_result")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        exner_w_implicit_weight_parameter: np.ndarray,
        theta_v_at_cells_on_half_levels: np.ndarray,
        ddqz_z_half: np.ndarray,
        tridiagonal_alpha_coeff_at_cells_on_half_levels: np.ndarray,
        tridiagonal_beta_coeff_at_cells_on_model_levels: np.ndarray,
        w_explicit_term: np.ndarray,
        exner_explicit_term: np.ndarray,
        tridiagonal_intermediate_result: np.ndarray,
        w: np.ndarray,
        dtime: float,
        cpd: float,
        **kwargs: Any,
    ) -> dict:
        z_q_ref, w_ref = solve_tridiagonal_matrix_for_w_forward_sweep_numpy(
            exner_w_implicit_weight_parameter,
            theta_v_at_cells_on_half_levels,
            ddqz_z_half,
            tridiagonal_alpha_coeff_at_cells_on_half_levels,
            tridiagonal_beta_coeff_at_cells_on_model_levels,
            exner_explicit_term,
            w_explicit_term,
            tridiagonal_intermediate_result,
            w,
            dtime,
            cpd,
        )
        return dict(tridiagonal_intermediate_result=z_q_ref, w=w_ref)

    @pytest.fixture
    def input_data(self, grid: base_grid.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        exner_w_implicit_weight_parameter = data_alloc.random_field(
            grid, dims.CellDim, dtype=ta.wpfloat
        )
        theta_v_at_cells_on_half_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat
        )
        ddqz_z_half = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        tridiagonal_alpha_coeff_at_cells_on_half_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.vpfloat
        )
        tridiagonal_beta_coeff_at_cells_on_model_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat
        )
        exner_explicit_term = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat
        )
        w_explicit_term = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
        )
        tridiagonal_intermediate_result = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat
        )
        # tridiagonal_intermediate_result first level should always be initialized to zero when solve_tridiagonal_matrix_for_w_forward_sweep is called
        tridiagonal_intermediate_result.asnumpy()[:, 0] = 0.0
        w = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        # w first level should always be initialized to zero when solve_tridiagonal_matrix_for_w_forward_sweep is called
        w.asnumpy()[:, 0] = 0.0

        h_start = 0
        h_end = gtx.int32(grid.num_cells)
        v_start = 1
        v_end = gtx.int32(grid.num_levels)

        return dict(
            exner_w_implicit_weight_parameter=exner_w_implicit_weight_parameter,
            theta_v_at_cells_on_half_levels=theta_v_at_cells_on_half_levels,
            ddqz_z_half=ddqz_z_half,
            tridiagonal_alpha_coeff_at_cells_on_half_levels=tridiagonal_alpha_coeff_at_cells_on_half_levels,
            tridiagonal_beta_coeff_at_cells_on_model_levels=tridiagonal_beta_coeff_at_cells_on_model_levels,
            w_explicit_term=w_explicit_term,
            exner_explicit_term=exner_explicit_term,
            tridiagonal_intermediate_result=tridiagonal_intermediate_result,
            w=w,
            dtime=ta.wpfloat("8.0"),
            cpd=ta.wpfloat("7.0"),
            horizontal_start=h_start,
            horizontal_end=h_end,
            vertical_start=v_start,
            vertical_end=v_end,
        )
