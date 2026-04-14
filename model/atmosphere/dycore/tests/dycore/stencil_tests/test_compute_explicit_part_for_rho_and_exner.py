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

from icon4py.model.atmosphere.dycore.stencils.compute_explicit_part_for_rho_and_exner import (
    compute_explicit_part_for_rho_and_exner,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


def compute_explicit_part_for_rho_and_exner_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    current_rho: np.ndarray,
    inv_ddqz_z_full: np.ndarray,
    divergence_of_mass: np.ndarray,
    vertical_mass_flux_at_cells_on_half_levels: np.ndarray,
    perturbed_exner_at_cells_on_model_levels: np.ndarray,
    tridiagonal_beta_coeff_at_cells_on_model_levels: np.ndarray,
    divergence_of_theta_v: np.ndarray,
    theta_v_at_cells_on_half_levels: np.ndarray,
    exner_tendency_due_to_slow_physics: np.ndarray,
    dtime: float,
) -> tuple[np.ndarray, np.ndarray]:
    rho_explicit_term = current_rho - dtime * inv_ddqz_z_full * (
        divergence_of_mass + vertical_mass_flux_at_cells_on_half_levels[:, :-1] - vertical_mass_flux_at_cells_on_half_levels[:, 1:]
    )

    exner_explicit_term = (
        perturbed_exner_at_cells_on_model_levels
        - tridiagonal_beta_coeff_at_cells_on_model_levels
        * (
            divergence_of_theta_v
            + (theta_v_at_cells_on_half_levels * vertical_mass_flux_at_cells_on_half_levels)[:, :-1]
            - (theta_v_at_cells_on_half_levels * vertical_mass_flux_at_cells_on_half_levels)[:, 1:]
        )
        + dtime * exner_tendency_due_to_slow_physics
    )
    return (rho_explicit_term, exner_explicit_term)


class TestComputeExplicitPartForRhoAndExner(StencilTest):
    PROGRAM = compute_explicit_part_for_rho_and_exner
    OUTPUTS = ("rho_explicit_term", "exner_explicit_term")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        current_rho: np.ndarray,
        inv_ddqz_z_full: np.ndarray,
        divergence_of_mass: np.ndarray,
        vertical_mass_flux_at_cells_on_half_levels: np.ndarray,
        perturbed_exner_at_cells_on_model_levels: np.ndarray,
        tridiagonal_beta_coeff_at_cells_on_model_levels: np.ndarray,
        divergence_of_theta_v: np.ndarray,
        theta_v_at_cells_on_half_levels: np.ndarray,
        exner_tendency_due_to_slow_physics: np.ndarray,
        dtime: float,
        **kwargs: Any,
    ) -> dict:
        (rho_explicit_term, exner_explicit_term) = compute_explicit_part_for_rho_and_exner_numpy(
            connectivities,
            current_rho=current_rho,
            inv_ddqz_z_full=inv_ddqz_z_full,
            divergence_of_mass=divergence_of_mass,
            vertical_mass_flux_at_cells_on_half_levels=vertical_mass_flux_at_cells_on_half_levels,
            perturbed_exner_at_cells_on_model_levels=perturbed_exner_at_cells_on_model_levels,
            tridiagonal_beta_coeff_at_cells_on_model_levels=tridiagonal_beta_coeff_at_cells_on_model_levels,
            divergence_of_theta_v=divergence_of_theta_v,
            theta_v_at_cells_on_half_levels=theta_v_at_cells_on_half_levels,
            exner_tendency_due_to_slow_physics=exner_tendency_due_to_slow_physics,
            dtime=dtime,
        )
        return dict(rho_explicit_term=rho_explicit_term, exner_explicit_term=exner_explicit_term)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        dtime = ta.wpfloat("1.0")
        current_rho = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        inv_ddqz_z_full = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        divergence_of_mass = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        vertical_mass_flux_at_cells_on_half_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
        )
        perturbed_exner_at_cells_on_model_levels = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        tridiagonal_beta_coeff_at_cells_on_model_levels = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        divergence_of_theta_v = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        theta_v_at_cells_on_half_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
        )
        exner_tendency_due_to_slow_physics = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)

        rho_explicit_term = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        exner_explicit_term = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)

        return dict(
            rho_explicit_term=rho_explicit_term,
            exner_explicit_term=exner_explicit_term,
            current_rho=current_rho,
            inv_ddqz_z_full=inv_ddqz_z_full,
            divergence_of_mass=divergence_of_mass,
            vertical_mass_flux_at_cells_on_half_levels=vertical_mass_flux_at_cells_on_half_levels,
            perturbed_exner_at_cells_on_model_levels=perturbed_exner_at_cells_on_model_levels,
            tridiagonal_beta_coeff_at_cells_on_model_levels=tridiagonal_beta_coeff_at_cells_on_model_levels,
            divergence_of_theta_v=divergence_of_theta_v,
            theta_v_at_cells_on_half_levels=theta_v_at_cells_on_half_levels,
            exner_tendency_due_to_slow_physics=exner_tendency_due_to_slow_physics,
            dtime=dtime,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
