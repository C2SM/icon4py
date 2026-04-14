# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any, Final

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.dycore.stencils.compute_results_for_thermodynamic_variables import (
    compute_results_for_thermodynamic_variables,
)
from icon4py.model.common import constants, dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


dycore_consts: Final = constants.PhysicsConstants()


def compute_results_for_thermodynamic_variables_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    rho_explicit_term: np.ndarray,
    exner_w_implicit_weight_parameter: np.ndarray,
    inv_ddqz_z_full: np.ndarray,
    rho_at_cells_on_half_levels: np.ndarray,
    w: np.ndarray,
    exner_explicit_term: np.ndarray,
    reference_exner_at_cells_on_model_levels: np.ndarray,
    tridiagonal_alpha_coeff_at_cells_on_half_levels: np.ndarray,
    tridiagonal_beta_coeff_at_cells_on_model_levels: np.ndarray,
    current_rho: np.ndarray,
    current_theta_v: np.ndarray,
    current_exner: np.ndarray,
    dtime: float,
) -> tuple[np.ndarray, ...]:
    rho_ic_offset_1 = rho_at_cells_on_half_levels[:, 1:]
    w_offset_0 = w[:, :-1]
    w_offset_1 = w[:, 1:]
    z_alpha_offset_1 = tridiagonal_alpha_coeff_at_cells_on_half_levels[:, 1:]
    exner_w_implicit_weight_parameter = np.expand_dims(exner_w_implicit_weight_parameter, axis=1)
    rho_new = rho_explicit_term - exner_w_implicit_weight_parameter * dtime * inv_ddqz_z_full * (
        rho_at_cells_on_half_levels[:, :-1] * w_offset_0 - rho_ic_offset_1 * w_offset_1
    )
    exner_new = (
        exner_explicit_term
        + reference_exner_at_cells_on_model_levels
        - tridiagonal_beta_coeff_at_cells_on_model_levels * (tridiagonal_alpha_coeff_at_cells_on_half_levels[:, :-1] * w_offset_0 - z_alpha_offset_1 * w_offset_1)
    )
    theta_v_new = (
        current_rho
        * current_theta_v
        * ((exner_new / current_exner - 1.0) * dycore_consts.cvd_o_rd + 1.0)
        / rho_new
    )
    return rho_new, exner_new, theta_v_new


class TestComputeResultsForThermodynamicVariables(StencilTest):
    PROGRAM = compute_results_for_thermodynamic_variables
    OUTPUTS = ("rho_new", "exner_new", "theta_v_new")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        rho_explicit_term: np.ndarray,
        exner_w_implicit_weight_parameter: np.ndarray,
        inv_ddqz_z_full: np.ndarray,
        rho_at_cells_on_half_levels: np.ndarray,
        w: np.ndarray,
        exner_explicit_term: np.ndarray,
        reference_exner_at_cells_on_model_levels: np.ndarray,
        tridiagonal_alpha_coeff_at_cells_on_half_levels: np.ndarray,
        tridiagonal_beta_coeff_at_cells_on_model_levels: np.ndarray,
        current_rho: np.ndarray,
        current_theta_v: np.ndarray,
        current_exner: np.ndarray,
        dtime: float,
        **kwargs: Any,
    ) -> dict:
        (rho_new, exner_new, theta_v_new) = compute_results_for_thermodynamic_variables_numpy(
            connectivities,
            rho_explicit_term=rho_explicit_term,
            exner_w_implicit_weight_parameter=exner_w_implicit_weight_parameter,
            inv_ddqz_z_full=inv_ddqz_z_full,
            rho_at_cells_on_half_levels=rho_at_cells_on_half_levels,
            w=w,
            exner_explicit_term=exner_explicit_term,
            reference_exner_at_cells_on_model_levels=reference_exner_at_cells_on_model_levels,
            tridiagonal_alpha_coeff_at_cells_on_half_levels=tridiagonal_alpha_coeff_at_cells_on_half_levels,
            tridiagonal_beta_coeff_at_cells_on_model_levels=tridiagonal_beta_coeff_at_cells_on_model_levels,
            current_rho=current_rho,
            current_theta_v=current_theta_v,
            current_exner=current_exner,
            dtime=dtime,
        )
        return dict(rho_new=rho_new, exner_new=exner_new, theta_v_new=theta_v_new)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        rho_explicit_term = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        exner_w_implicit_weight_parameter = data_alloc.random_field(grid, dims.CellDim, dtype=ta.wpfloat)
        inv_ddqz_z_full = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        rho_at_cells_on_half_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
        )
        w = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
        )
        exner_explicit_term = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        reference_exner_at_cells_on_model_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat
        )
        tridiagonal_alpha_coeff_at_cells_on_half_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.vpfloat
        )
        tridiagonal_beta_coeff_at_cells_on_model_levels = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        current_rho = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        current_theta_v = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        current_exner = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        rho_new = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        exner_new = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        theta_v_new = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        dtime = ta.wpfloat("5.0")

        return dict(
            rho_explicit_term=rho_explicit_term,
            exner_w_implicit_weight_parameter=exner_w_implicit_weight_parameter,
            inv_ddqz_z_full=inv_ddqz_z_full,
            rho_at_cells_on_half_levels=rho_at_cells_on_half_levels,
            w=w,
            exner_explicit_term=exner_explicit_term,
            reference_exner_at_cells_on_model_levels=reference_exner_at_cells_on_model_levels,
            tridiagonal_alpha_coeff_at_cells_on_half_levels=tridiagonal_alpha_coeff_at_cells_on_half_levels,
            tridiagonal_beta_coeff_at_cells_on_model_levels=tridiagonal_beta_coeff_at_cells_on_model_levels,
            current_rho=current_rho,
            current_theta_v=current_theta_v,
            current_exner=current_exner,
            rho_new=rho_new,
            exner_new=exner_new,
            theta_v_new=theta_v_new,
            dtime=dtime,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
